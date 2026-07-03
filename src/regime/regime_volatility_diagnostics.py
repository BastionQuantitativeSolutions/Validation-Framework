from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _safe_json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _hash_json_file(path: Path) -> Optional[str]:
    payload = _safe_json_load(path)
    if payload is None:
        return None
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_bytes(path: Path) -> Optional[str]:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    return hashlib.sha256(data).hexdigest()


def _find_line_matches(path: Path, patterns: Sequence[str]) -> List[Dict[str, Any]]:
    text = _read_text(path)
    if not text:
        return []
    lines = text.splitlines()
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(lines, start=1):
        if any(p in line for p in patterns):
            out.append({"file": str(path), "line": i, "snippet": line.strip()[:300]})
    return out


def _normalize_regime_source(source: str) -> str:
    source_up = str(source or "").strip().lower()
    if source_up in {"trained_cache", "trainedcache"}:
        return "trained_cache"
    if source_up in {
        "trained_cache+live_update",
        "trained_cache_live_update",
        "trained_cache+liveupdate",
    }:
        return "trained_cache+live_update"
    if source_up in {"detector", "live", "runtime_detector"}:
        return "live_recompute"
    if source_up in {"trained", "training"}:
        return "trained_cache"
    if source_up in {"cache", "cached"}:
        return "trained_cache"
    if source_up in {"smc"}:
        return "live_recompute"
    return "live_recompute"


@dataclass
class HashSnapshot:
    cluster_hash: Optional[str] = None
    regime_hash: Optional[str] = None
    taken_at: Optional[str] = None


class RegimeVolatilityDiagnostics:
    """Observability-only diagnostics for startup training vs live runtime usage."""

    def __init__(self, results_dir: Path, data_models_results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self.data_models_results_dir = Path(data_models_results_dir)
        self.explainability_dir = self.results_dir / "EXPLAINABILITY"
        self.explainability_dir.mkdir(parents=True, exist_ok=True)

        self.repo_root = Path(__file__).resolve().parents[3]
        self.core_root = self.repo_root / "CORE_MODULES"
        self.runtime_file = self.core_root / "core" / "runtime" / "ensemble_all_in_one.py"
        self.regime_file = self.core_root / "core" / "regime" / "market_regime.py"
        self.hook_file = self.core_root / "core" / "regime" / "volatility_regime_hook.py"
        self.preflight_file = self.core_root / "core" / "monitoring" / "pre_flight.py"

        self.volatility_cache_path = self.data_models_results_dir / "volatility_clusters.json"
        self.regime_training_cache_path = self.data_models_results_dir / "regime_training_cache.json"
        self.runtime_regime_cache_path = self.results_dir / "regime_cache.json"

        self.dataflow_map_path = self.explainability_dir / "regime_volatility_dataflow_map.json"
        self.cache_report_path = self.explainability_dir / "regime_cache_integrity_report.json"
        self.override_audit_path = self.explainability_dir / "regime_runtime_override_audit.json"
        self.live_behavior_path = self.explainability_dir / "regime_live_update_behavior.json"
        self.final_diagnosis_path = self.explainability_dir / "regime_volatility_final_diagnosis.json"

        self.training_invocation_count = 0
        self.training_completed = False
        self.startup_snapshot = HashSnapshot()
        self.first_cycle_snapshot = HashSnapshot()
        self.fifth_cycle_snapshot = HashSnapshot()
        self.latest_snapshot = HashSnapshot()

        self.current_cycle_index = 0
        self.current_cycle_logged = False
        self.last_detector_id: Optional[int] = None
        self.detector_reinstantiated = False
        self.detector_ids_by_cycle: Dict[str, List[int]] = {}
        self.regime_sources_counter: Counter = Counter()
        self.last_regime_source = "live_recompute"
        self.cache_loaded = self.volatility_cache_path.exists() and self.regime_training_cache_path.exists()
        self.observed_using_trained_instance = False
        self.current_detector_instance_id: Optional[int] = None
        self.current_runtime_cache_hash: Optional[str] = None
        self.current_live_update_applied = False
        self.detector_instance_id_startup: Optional[int] = None
        self.detector_instance_id_cycle5: Optional[int] = None

        self.runtime_uses_training_state = self._runtime_references_training_state()
        self.per_cycle_recompute_overwrites = self._per_cycle_recompute_detected()
        self.fallback_paths = self._scan_fallback_paths()
        self.override_points = self._scan_override_points()

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        except Exception as e:
            log.warning(f"[REGIME_DIAG] Failed to write {path}: {e}")

    def _runtime_references_training_state(self) -> bool:
        # Runtime integration is considered "trained-state referenced" if the detector supports
        # trained-state binding and runtime invokes that binding path.
        runtime_text = _read_text(self.runtime_file)
        regime_text = _read_text(self.regime_file)

        runtime_bind = ("bind_trained_state(" in runtime_text) and ("initialize_volatility_and_regime(" in runtime_text)
        detector_bind = ("def bind_trained_state(" in regime_text) and ("using_trained_instance" in regime_text)
        return bool(runtime_bind and detector_bind)

    def _per_cycle_recompute_detected(self) -> bool:
        text = _read_text(self.regime_file)
        if not text:
            return False
        has_live_fetch = "copy_rates_from_pos" in text
        has_cache_write = "self._save_cache()" in text
        return bool(has_live_fetch and has_cache_write)

    def _scan_fallback_paths(self) -> List[Dict[str, Any]]:
        return _find_line_matches(
            self.regime_file,
            [
                "return self._get_default_regime()",
                "if mt5 is None",
                "len(bars) < 100",
                "_get_default_regime(",
            ],
        )

    def _scan_override_points(self) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        points.extend(
            _find_line_matches(
                self.runtime_file,
                [
                    "get_regime_detector(",
                    "detect_regime(",
                    "initialize_volatility_and_regime(",
                ],
            )
        )
        points.extend(
            _find_line_matches(
                self.regime_file,
                [
                    "copy_rates_from_pos",
                    "self._save_cache()",
                    "_load_cache(",
                    "ENABLE_PARQUET_REGIME",
                ],
            )
        )
        points.extend(
            _find_line_matches(
                self.hook_file,
                [
                    "train_volatility_clustering(",
                    "train_regime_detection(",
                    "Loaded cached volatility clusters",
                    "Loaded cached regime states",
                    "Saved volatility clusters to cache",
                    "Saved regime states to cache",
                ],
            )
        )
        return points

    def _build_dataflow_map(self) -> Dict[str, Any]:
        training_entry = []
        training_entry.extend(_find_line_matches(self.runtime_file, ["initialize_volatility_and_regime("]))
        training_entry.extend(_find_line_matches(self.preflight_file, ["initialize_volatility_and_regime("]))

        cache_writes = _find_line_matches(
            self.hook_file,
            [
                "volatility_clusters.json",
                "regime_training_cache.json",
                "Saved volatility clusters to cache",
                "Saved regime states to cache",
            ],
        )
        cache_loads = []
        cache_loads.extend(
            _find_line_matches(
                self.hook_file,
                [
                    "Loaded cached volatility clusters",
                    "Loaded cached regime states",
                ],
            )
        )
        cache_loads.extend(
            _find_line_matches(
                self.regime_file,
                [
                    "self._load_cache()",
                    "regime_cache.json",
                ],
            )
        )

        live_calls = _find_line_matches(self.runtime_file, ["detect_regime("])
        live_detector_impl = _find_line_matches(self.regime_file, ["copy_rates_from_pos", "self._save_cache()"])

        runtime_mode = (
            "trained_cache+live_update"
            if (
                self.observed_using_trained_instance
                or "trained_cache+live_update" in self.regime_sources_counter
                or "trained_cache" in self.regime_sources_counter
            )
            else "live_recompute_mt5_recent_window"
        )

        source_of_truth = {
            "startup_training": {
                "volatility": str(self.volatility_cache_path),
                "regime": str(self.regime_training_cache_path),
            },
            "runtime_regime_classification": runtime_mode,
            "runtime_volatility_classification": "detector_volatility_percentile_or_smc_fallback",
            "runtime_cache": str(self.runtime_regime_cache_path),
        }

        dataflow = {
            "generated_at": _utc_now_iso(),
            "source_of_truth": source_of_truth,
            "startup_training_entry_points": training_entry,
            "cache_write_locations": cache_writes,
            "cache_load_locations": cache_loads,
            "live_regime_detection_calls": live_calls,
            "live_detector_implementation_points": live_detector_impl,
            "pipeline_map": [
                "22M training -> DATA_MODELS/results/volatility_clusters.json + regime_training_cache.json",
                "runtime detector init -> CORE_MODULES/results/regime_cache.json loader",
                "per-cycle signal context -> regime_detector.detect_regime(pair, tf)",
                "governor context consumes cycle regime outputs (regime/regime_confidence/volatility_percentile)",
            ],
            "runtime_detector_references_cached_trained_states": bool(self.runtime_uses_training_state or self.observed_using_trained_instance),
            "per_cycle_recomputation_overwrites_runtime_cache": bool(self.per_cycle_recompute_overwrites),
            "training_cache_consumed_in_signal_path": bool(self.runtime_uses_training_state or self.observed_using_trained_instance),
            "observed_regime_source_distribution": dict(self.regime_sources_counter),
        }
        return dataflow

    def bootstrap_static_reports(self) -> None:
        dataflow = self._build_dataflow_map()
        self._write_json(self.dataflow_map_path, dataflow)
        self._write_override_audit()
        self._write_live_behavior()
        self._write_final_diagnosis()

    def note_training_invocation(self) -> None:
        self.training_invocation_count += 1
        self._write_override_audit()
        self._write_final_diagnosis()

    def record_startup_training(self, training_succeeded: bool) -> None:
        self.training_completed = bool(training_succeeded)
        self.cache_loaded = self.volatility_cache_path.exists() and self.regime_training_cache_path.exists()
        self.startup_snapshot = self._capture_hash_snapshot()
        self.latest_snapshot = self.startup_snapshot
        self._write_cache_integrity_report()
        self._write_live_behavior()
        self._write_override_audit()
        self._write_final_diagnosis()

    def _capture_hash_snapshot(self) -> HashSnapshot:
        return HashSnapshot(
            cluster_hash=_hash_json_file(self.volatility_cache_path),
            regime_hash=_hash_json_file(self.regime_training_cache_path),
            taken_at=_utc_now_iso(),
        )

    def record_cycle_start(self, detector: Any) -> None:
        self.current_cycle_index += 1
        self.current_cycle_logged = False
        cycle_key = str(self.current_cycle_index)

        det_id = id(detector) if detector is not None else 0
        self.current_detector_instance_id = det_id or None
        ids = self.detector_ids_by_cycle.setdefault(cycle_key, [])
        if det_id and det_id not in ids:
            ids.append(det_id)

        if self.last_detector_id is not None and det_id and det_id != self.last_detector_id:
            self.detector_reinstantiated = True
        if det_id:
            self.last_detector_id = det_id
        if self.current_cycle_index == 1:
            self.detector_instance_id_startup = det_id or None
        if self.current_cycle_index == 5:
            self.detector_instance_id_cycle5 = det_id or None

        if detector is not None:
            try:
                using_trained = bool(getattr(detector, "using_trained_instance", False))
            except Exception:
                using_trained = False
            self.observed_using_trained_instance = self.observed_using_trained_instance or using_trained
            self.current_live_update_applied = bool(getattr(detector, "last_live_update_applied", False))
            if hasattr(detector, "get_runtime_cache_hash"):
                try:
                    self.current_runtime_cache_hash = detector.get_runtime_cache_hash()
                except Exception:
                    self.current_runtime_cache_hash = None

        snap = self._capture_hash_snapshot()
        self.latest_snapshot = snap
        if self.current_cycle_index == 1:
            self.first_cycle_snapshot = snap
        if self.current_cycle_index == 5:
            self.fifth_cycle_snapshot = snap

        self._write_cache_integrity_report()
        self._write_override_audit()
        self._write_live_behavior()
        self._write_final_diagnosis()

    def note_regime_source(self, regime_source: str) -> None:
        mapped_source = _normalize_regime_source(regime_source)
        self.last_regime_source = mapped_source
        self.regime_sources_counter[mapped_source] += 1
        if mapped_source in {"trained_cache", "trained_cache+live_update"}:
            self.observed_using_trained_instance = True
        if mapped_source == "trained_cache+live_update":
            self.current_live_update_applied = True

        if self.current_cycle_logged:
            return

        using_trained_model = bool(self.observed_using_trained_instance or self.runtime_uses_training_state)
        centroids_changed = (
            self.startup_snapshot.cluster_hash is not None
            and self.latest_snapshot.cluster_hash is not None
            and self.startup_snapshot.cluster_hash != self.latest_snapshot.cluster_hash
        )
        log.info(f"[REGIME_DIAG] using_trained_model={using_trained_model}")
        log.info(f"[REGIME_DIAG] cache_loaded={self.cache_loaded}")
        log.info(f"[REGIME_DIAG] detector_reinstantiated={self.detector_reinstantiated}")
        log.info(f"[REGIME_DIAG] centroids_changed={centroids_changed}")
        log.info(f"[REGIME_DIAG] regime_source={mapped_source}")
        self.current_cycle_logged = True

    def _write_cache_integrity_report(self) -> None:
        cluster_hash_startup = self.startup_snapshot.cluster_hash
        cluster_hash_runtime = self.first_cycle_snapshot.cluster_hash
        cluster_hash_cycle5 = self.fifth_cycle_snapshot.cluster_hash
        regime_hash_startup = self.startup_snapshot.regime_hash
        regime_hash_runtime = self.first_cycle_snapshot.regime_hash
        regime_hash_cycle5 = self.fifth_cycle_snapshot.regime_hash

        mismatch_detected = False
        if cluster_hash_startup and cluster_hash_runtime and cluster_hash_startup != cluster_hash_runtime:
            mismatch_detected = True
        if regime_hash_startup and regime_hash_runtime and regime_hash_startup != regime_hash_runtime:
            mismatch_detected = True

        payload = {
            "generated_at": _utc_now_iso(),
            "cluster_hash_startup": cluster_hash_startup,
            "cluster_hash_runtime": cluster_hash_runtime,
            "cluster_hash_after_5_cycles": cluster_hash_cycle5,
            "regime_hash_startup": regime_hash_startup,
            "regime_hash_runtime": regime_hash_runtime,
            "regime_hash_after_5_cycles": regime_hash_cycle5,
            "startup_snapshot_at": self.startup_snapshot.taken_at,
            "runtime_first_cycle_at": self.first_cycle_snapshot.taken_at,
            "runtime_fifth_cycle_at": self.fifth_cycle_snapshot.taken_at,
            "runtime_regime_cache_hash": _hash_json_file(self.runtime_regime_cache_path),
            "runtime_detector_instance_id_cycle1": self.detector_instance_id_startup,
            "runtime_detector_instance_id_cycle5": self.detector_instance_id_cycle5,
            "runtime_detector_instance_changed": (
                self.detector_instance_id_startup is not None
                and self.detector_instance_id_cycle5 is not None
                and self.detector_instance_id_startup != self.detector_instance_id_cycle5
            ),
            "mismatch_detected": mismatch_detected,
            "cache_files_present": {
                "volatility_clusters": self.volatility_cache_path.exists(),
                "regime_training_cache": self.regime_training_cache_path.exists(),
                "runtime_regime_cache": self.runtime_regime_cache_path.exists(),
            },
            "notes": {
                "training_cache_hash_source": "DATA_MODELS/results caches",
                "runtime_hash_source": "first and fifth live cycle checkpoints",
            },
        }
        self._write_json(self.cache_report_path, payload)

    def _write_override_audit(self) -> None:
        detector_instances_per_cycle = {cycle: len(ids) for cycle, ids in sorted(self.detector_ids_by_cycle.items(), key=lambda kv: int(kv[0]))}
        payload = {
            "generated_at": _utc_now_iso(),
            "override_points": self.override_points,
            "fallback_paths": self.fallback_paths,
            "detector_instances_per_cycle": detector_instances_per_cycle,
            "detector_instance_id_cycle1": self.detector_instance_id_startup,
            "detector_instance_id_cycle5": self.detector_instance_id_cycle5,
            "detector_instance_changed_by_cycle5": (
                self.detector_instance_id_startup is not None
                and self.detector_instance_id_cycle5 is not None
                and self.detector_instance_id_startup != self.detector_instance_id_cycle5
            ),
            "duplicate_training_detected": bool(self.training_invocation_count > 1),
            "training_invocation_count_runtime": int(self.training_invocation_count),
            "detector_reinstantiated": bool(self.detector_reinstantiated),
            "runtime_recompute_detected": bool(self.per_cycle_recompute_overwrites),
            "observed_using_trained_instance": bool(self.observed_using_trained_instance),
        }
        self._write_json(self.override_audit_path, payload)

    def _write_live_behavior(self) -> None:
        centroid_drift = {
            "startup_vs_first_cycle": (
                self.startup_snapshot.cluster_hash != self.first_cycle_snapshot.cluster_hash
                if self.startup_snapshot.cluster_hash and self.first_cycle_snapshot.cluster_hash
                else None
            ),
            "startup_vs_latest": (
                self.startup_snapshot.cluster_hash != self.latest_snapshot.cluster_hash
                if self.startup_snapshot.cluster_hash and self.latest_snapshot.cluster_hash
                else None
            ),
            "runtime_regime_cache_hash": _hash_json_file(self.runtime_regime_cache_path),
        }

        live_updates_incremental = bool(self.current_live_update_applied or ("trained_cache+live_update" in self.regime_sources_counter))
        using_trained = bool(self.observed_using_trained_instance or self.runtime_uses_training_state)
        model_type = "trained_cache+incremental_live_update" if using_trained else "recalculated_live_window"
        confidence_source = "trained_cache+rolling_live_bars" if using_trained else "rolling_live_bars"

        payload = {
            "generated_at": _utc_now_iso(),
            "model_type": model_type,
            "cluster_centroid_drift": centroid_drift,
            "regime_confidence_source": confidence_source,
            "using_trained_parameters_as_base": using_trained,
            "live_updates_incremental": live_updates_incremental,
            "runtime_overwrites_regime_cache": bool(self.per_cycle_recompute_overwrites),
            "observed_regime_source_distribution": dict(self.regime_sources_counter),
            "last_regime_source": self.last_regime_source,
            "live_update_applied_last_cycle": bool(self.current_live_update_applied),
        }
        self._write_json(self.live_behavior_path, payload)

    def _write_final_diagnosis(self) -> None:
        runtime_using_trained_clusters = bool(
            self.observed_using_trained_instance
            or ("trained_cache" in self.regime_sources_counter)
            or ("trained_cache+live_update" in self.regime_sources_counter)
            or self.runtime_uses_training_state
        )
        startup_training_effective = bool(self.training_completed and runtime_using_trained_clusters)
        live_updates_incremental = bool(self.current_live_update_applied or ("trained_cache+live_update" in self.regime_sources_counter))
        unintended_retraining = bool(self.training_invocation_count > 1)
        cache_shadowing_detected = bool(self.training_completed and (not runtime_using_trained_clusters) and self.per_cycle_recompute_overwrites)

        evidence_flags = [
            1.0 if startup_training_effective else 0.0,
            1.0 if runtime_using_trained_clusters else 0.0,
            1.0 if live_updates_incremental else 0.0,
            0.0 if unintended_retraining else 1.0,
            0.0 if cache_shadowing_detected else 1.0,
        ]
        confidence_score = round(sum(evidence_flags) / len(evidence_flags), 4)

        payload = {
            "generated_at": _utc_now_iso(),
            "startup_training_effective": startup_training_effective,
            "runtime_using_trained_clusters": runtime_using_trained_clusters,
            "live_updates_incremental": live_updates_incremental,
            "unintended_retraining": unintended_retraining,
            "cache_shadowing_detected": cache_shadowing_detected,
            "confidence_score": confidence_score,
            "evidence": {
                "training_invocation_count_runtime": self.training_invocation_count,
                "training_completed": self.training_completed,
                "runtime_recompute_detected": self.per_cycle_recompute_overwrites,
                "detector_reinstantiated": self.detector_reinstantiated,
                "last_regime_source": self.last_regime_source,
                "observed_using_trained_instance": self.observed_using_trained_instance,
                "current_live_update_applied": self.current_live_update_applied,
            },
        }
        self._write_json(self.final_diagnosis_path, payload)
