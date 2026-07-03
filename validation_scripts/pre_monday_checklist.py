"""
pre_monday_checklist.py
=======================
Cavalier Bastion V4.0 — Pre-Live-Trading Readiness Checker
Run this BEFORE resuming live trading to confirm all systems are healthy.

Exit codes:
    0 = ALL CLEAR  — safe to launch live trading
    1 = WARNINGS   — review flagged items; trading may proceed with caution
    2 = BLOCKED    — critical failures; do NOT start live trading

Usage:
    python pre_monday_checklist.py
    python pre_monday_checklist.py --verbose
    python pre_monday_checklist.py --skip-mt5   (CI / offline environments)
"""

import os
import sys
import json
import time
import shutil
import argparse
import datetime
import platform
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Root detection (mirrors validate_v10_5_0.py logic)
# ---------------------------------------------------------------------------
def _find_root() -> Path:
    env_root = os.getenv("CAVALIER_ROOT") or os.getenv("PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).resolve()
        if (candidate / "DATA_MODELS" / "data_parquet").exists():
            return candidate
    for start in [Path(__file__).resolve().parent, Path.cwd()]:
        for parent in [start] + list(start.parents):
            if (
                (parent / "DATA_MODELS" / "data_parquet").exists()
                and (parent / "CORE_MODULES").exists()
            ):
                return parent
    # Hard fallback: pre_monday_checklist.py → validation/ → CORE_MODULES/ → ROOT
    return Path(__file__).resolve().parents[2]

BASE_DIR = _find_root()
CORE_DIR = BASE_DIR / "CORE_MODULES"

# ---------------------------------------------------------------------------
# Canonical paths — all derived from actual system layout
# ---------------------------------------------------------------------------
DATA_DIR    = BASE_DIR / "DATA_MODELS" / "data_parquet"
MODEL_DIR   = BASE_DIR / "DATA_MODELS" / "models_live"
RESULTS_DIR = CORE_DIR / "results"
LOGS_DIR    = BASE_DIR / "logs"
CONFIG_DIR  = CORE_DIR / "config"
VALIDATION_DIR = CORE_DIR / "validation"
LOCK_FILE   = BASE_DIR / ".cavalier_instance.lock"

# Config files — authoritative locations confirmed from system scan
RISK_GOV    = CONFIG_DIR / "risk_governor.json"
FTMO_STATE_CANDIDATES = [
    CONFIG_DIR / "ftmo_state.json",
    CORE_DIR / "results" / "ftmo_state.json",
    CORE_DIR / "data" / "ftmo_state.json",
]

# ---------------------------------------------------------------------------
# Canonical pair/timeframe lists — MUST match pre_flight.py and constants.py
# ---------------------------------------------------------------------------
ALL_PAIRS = [
    "EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "AUDUSD",
    "USDCAD", "USDCHF", "GBPCHF", "XAGUSD", "NZDUSD",
    "USOIL",  "UKOIL",  "US100",  "JP225",  "HK50",
    "UK100",  "BTCUSD", "ETHUSD",
]
QUARANTINED_PAIRS = {"USDJPY", "GBPCHF", "NZDUSD", "UKOIL"}
LIVE_PAIRS  = [p for p in ALL_PAIRS if p not in QUARANTINED_PAIRS]

# Live TFS — from pre_flight.py (H4 excluded from live config)
REQUIRED_TFS    = ["M1", "M5", "M15", "M30", "H1", "D1"]
# Primary model TFs (these MUST have models; D1/M1 are bonus)
CORE_MODEL_TFS  = ["M5", "M15", "M30", "H1"]

# System scripts that must exist (correct paths from codebase scan)
REQUIRED_SCRIPTS = {
    "main_loop.py":      CORE_DIR / "core" / "runtime" / "main_loop.py",
    "risk_governor.py":  CORE_DIR / "core" / "risk" / "risk_governor.py",
    "smc_confluence.py": CORE_DIR / "core" / "smc" / "smc_confluence.py",
}

MIN_DISK_GB     = 5.0
MIN_DATA_ROWS   = 5_000
STALE_LOCK_SECONDS = 3600

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
if platform.system() == "Windows":
    os.system("color")

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _ok(msg):   return f"  {GREEN}✓{RESET}  {msg}"
def _warn(msg): return f"  {YELLOW}⚠{RESET}  {msg}"
def _fail(msg): return f"  {RED}✗{RESET}  {msg}"

# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------
class CheckResult:
    def __init__(self, name: str):
        self.name = name
        self.passed: List[str] = []
        self.warnings: List[str] = []
        self.failures: List[str] = []

    def ok(self, msg):   self.passed.append(msg)
    def warn(self, msg): self.warnings.append(msg)
    def fail(self, msg): self.failures.append(msg)

    @property
    def status(self):
        if self.failures:
            return "FAIL"
        if self.warnings:
            return "WARN"
        return "PASS"

    def print_section(self, verbose=False):
        colour = GREEN if self.status == "PASS" else (YELLOW if self.status == "WARN" else RED)
        print(f"\n{BOLD}{colour}[{self.status}]{RESET} {BOLD}{self.name}{RESET}")
        for msg in self.failures:
            print(_fail(msg))
        for msg in self.warnings:
            print(_warn(msg))
        if verbose:
            for msg in self.passed:
                print(_ok(msg))
        elif not self.failures and not self.warnings:
            print(_ok(f"All {len(self.passed)} checks passed"))


# ===========================================================================
# CHECK FUNCTIONS
# ===========================================================================

def check_environment(verbose: bool) -> CheckResult:
    r = CheckResult("Environment & Dependencies")

    major, minor = sys.version_info.major, sys.version_info.minor
    if major == 3 and minor >= 9:
        r.ok(f"Python {major}.{minor}")
    else:
        r.fail(f"Python {major}.{minor} — requires 3.9+")

    packages = {
        "numpy":         "numpy",
        "pandas":        "pandas",
        "sklearn":       "scikit-learn",
        "xgboost":       "xgboost",
        "lightgbm":      "lightgbm",
        "catboost":      "catboost",
        "joblib":        "joblib",
        "MetaTrader5":   "MetaTrader5",
    }
    for imp, pip in packages.items():
        try:
            __import__(imp)
            r.ok(f"{pip} importable")
        except ImportError:
            r.fail(f"{pip} NOT installed")

    omp = os.environ.get("OMP_NUM_THREADS", "not set")
    if omp == "1":
        r.ok("OMP_NUM_THREADS=1 (GBM thread-safety)")
    else:
        r.warn(f"OMP_NUM_THREADS={omp} — set to '1' to prevent 0xC06D007F crash")

    r.ok(f"ROOT resolved: {BASE_DIR}")
    for d in [BASE_DIR, CORE_DIR, DATA_DIR, MODEL_DIR]:
        if d.exists():
            r.ok(f"Directory present: {d.relative_to(BASE_DIR)}/")
        else:
            r.fail(f"Missing directory: {d}")

    return r


def check_disk_space(verbose: bool) -> CheckResult:
    r = CheckResult("Disk Space & Write Permissions")

    usage = shutil.disk_usage(BASE_DIR)
    free_gb  = usage.free  / (1024**3)
    total_gb = usage.total / (1024**3)
    if free_gb >= MIN_DISK_GB:
        r.ok(f"{free_gb:.1f} GB free of {total_gb:.1f} GB")
    elif free_gb >= 2.0:
        r.warn(f"Only {free_gb:.1f} GB free — recommend ≥{MIN_DISK_GB} GB")
    else:
        r.fail(f"Critical: only {free_gb:.1f} GB free")

    for d, label in [(RESULTS_DIR, "results"), (LOGS_DIR, "logs")]:
        d.mkdir(parents=True, exist_ok=True)
        test = d / ".write_test"
        try:
            test.write_text("ok")
            test.unlink()
            r.ok(f"Write OK: {label}/")
        except Exception as e:
            if label == "results":
                r.fail(f"Cannot write to {label}/: {e}")
            else:
                r.warn(f"Cannot write to {label}/: {e}")

    return r


def check_data_files(verbose: bool) -> CheckResult:
    r = CheckResult("Historical Data Files (Parquet)")

    try:
        import pandas as pd
    except ImportError:
        r.fail("pandas not installed — cannot check parquet files")
        return r

    missing, thin, valid = [], [], []

    for pair in LIVE_PAIRS:
        for tf in REQUIRED_TFS:
            path = DATA_DIR / f"{pair}_{tf}.parquet"
            if tf == "M1" and not path.exists():
                live_path = DATA_DIR / f"{pair}_{tf}_LIVE.parquet"
                if live_path.exists():
                    path = live_path
            if not path.exists():
                missing.append(f"{pair}/{tf}")
            else:
                try:
                    df = pd.read_parquet(path, columns=["close"])
                    n = len(df)
                    if n < MIN_DATA_ROWS:
                        thin.append(f"{pair}/{tf} ({n:,} rows)")
                    else:
                        valid.append(f"{pair}/{tf} ({n:,} rows)")
                except Exception as e:
                    thin.append(f"{pair}/{tf} — read error: {e}")

    for v in valid:
        if verbose:
            r.ok(v)
    for t in thin:
        r.warn(t)
    for m in missing:
        r.fail(f"Missing data: {m}")

    summary = f"{len(valid)} files valid, {len(thin)} thin, {len(missing)} missing"
    if missing:
        r.fail(f"Data coverage: {summary}")
    elif thin:
        r.warn(f"Data coverage: {summary}")
    else:
        r.ok(f"Data coverage: {summary}")

    return r


def check_models(verbose: bool) -> CheckResult:
    r = CheckResult("ML Ensemble Models (DATA_MODELS/models_live)")

    try:
        import joblib  # noqa: F401
    except ImportError:
        r.fail("joblib not installed — cannot verify models")
        return r

    missing_pairs, present_pairs = [], []

    model_roots = [MODEL_DIR / "models_all", MODEL_DIR]
    model_markers = [
        "lgb/lgb_long_hit_1_3R.txt",
        "xgb/xgb_long_hit_1_3R.json",
        "cat/cat_long_hit_1_3R.cbm",
        "scaler.pkl",
    ]

    for pair in LIVE_PAIRS:
        pair_found = False
        for tf in CORE_MODEL_TFS:
            model_base = next(
                (root / f"{pair}_{tf}" for root in model_roots if (root / f"{pair}_{tf}").exists()),
                None,
            )
            if model_base is None or not model_base.exists():
                continue
            models = [model_base / marker for marker in model_markers if (model_base / marker).exists()]
            if models:
                pair_found = True
                if verbose:
                    r.ok(f"{pair}/{tf}: MLS V3 model artifact(s) present")
                break  # at least one TF has models — pair is covered

        if pair_found:
            present_pairs.append(pair)
        else:
            # Check if any TF has ANY model at all
            any_tf_dir = next(
                (root / f"{pair}_M5" for root in model_roots if (root / f"{pair}_M5").exists()),
                None,
            )
            if any_tf_dir is None or not any_tf_dir.exists():
                missing_pairs.append(pair)
            else:
                r.warn(f"{pair}: model directory exists but no recognized live model artifact found")

    for p in present_pairs:
        r.ok(f"{p}: models present")
    for p in missing_pairs:
        r.warn(f"{p}: no model directory in models_live/ — will use proxy fallback")

    # Count total model shards
    total_shards = sum(
        len(list((MODEL_DIR / "models_all").rglob(pattern)))
        for pattern in ("lgb_*.txt", "xgb_*.json", "cat_*.cbm")
    )
    r.ok(f"Total MLS V3 model artifacts found: {total_shards:,}")

    # FAISS index
    faiss_shards_dir = BASE_DIR / "CORE_MODULES" / "results" / "LLAMA_CACHE" / "rag_database" / "faiss_shards"
    faiss_files = (
        list(MODEL_DIR.rglob("*.faiss")) 
        + list(BASE_DIR.rglob("*.faiss"))
        + list(MODEL_DIR.rglob("*.faissindex"))
        + list(BASE_DIR.rglob("*.faissindex"))
        + list(faiss_shards_dir.rglob("*.faissindex"))
    )
    # Remove duplicates
    faiss_files = list(set(faiss_files))
    if faiss_files:
        for fi in sorted(faiss_files)[:5]:
            size_mb = fi.stat().st_size / (1024**2)
            r.ok(f"FAISS: {fi.name} ({size_mb:.1f} MB)")
    else:
        r.warn("No FAISS *.faiss/*.faissindex index files found — RAG system will cold-start on launch")

    return r


def check_risk_config(verbose: bool) -> CheckResult:
    r = CheckResult("Risk Configuration & FTMO State")

    # risk_governor.json
    if RISK_GOV.exists():
        try:
            with open(RISK_GOV) as f:
                rg = json.load(f)
            r.ok(f"risk_governor.json readable (v{rg.get('version','?')})")

            # Validate v10.5.0 entry thresholds
            et = rg.get("entry_thresholds", {})
            bt = et.get("buy_threshold")
            st = et.get("sell_threshold")
            if bt is not None:
                if 0.52 <= float(bt) <= 0.54:
                    r.ok(f"  buy_threshold = {bt} (v10.5.0 ✓)")
                else:
                    r.fail(f"  buy_threshold = {bt} — expected ~0.53 (v10.5.0)")
            else:
                r.warn("  entry_thresholds.buy_threshold not found")

            if st is not None:
                if 0.46 <= float(st) <= 0.48:
                    r.ok(f"  sell_threshold = {st} (v10.5.0 ✓)")
                else:
                    r.fail(f"  sell_threshold = {st} — expected ~0.47 (v10.5.0)")
            else:
                r.warn("  entry_thresholds.sell_threshold not found")

            # Max concurrent positions
            mcp = rg.get("max_concurrent_positions", 0)
            if mcp >= 10:
                r.ok(f"  max_concurrent_positions = {mcp}")
            else:
                r.warn(f"  max_concurrent_positions = {mcp} (low — expected ~50)")

            # Circuit breaker state
            cb = rg.get("circuit_breaker", {})
            cb_state = cb.get("state", "CLOSED")  # default to CLOSED if not set
            if cb_state in ("CLOSED", "closed", None, ""):
                r.ok("  Circuit breaker: CLOSED")
            elif cb_state in ("HALF_OPEN",):
                r.warn("  Circuit breaker: HALF_OPEN — monitor after start")
            elif cb_state in ("OPEN",):
                r.fail("  Circuit breaker: OPEN — system will not trade")
            else:
                r.ok(f"  Circuit breaker state: {cb_state}")

        except json.JSONDecodeError as e:
            r.fail(f"risk_governor.json parse error: {e}")
        except Exception as e:
            r.fail(f"risk_governor.json error: {e}")
    else:
        r.warn(f"risk_governor.json not found at {RISK_GOV} — will use defaults")

    # ftmo_state.json — check all candidate paths
    ftmo_path = next((p for p in FTMO_STATE_CANDIDATES if p.exists()), None)
    if ftmo_path:
        try:
            with open(ftmo_path) as f:
                ftmo = json.load(f)
            r.ok(f"ftmo_state.json readable ({ftmo_path.parent.name}/)")

            blocked = ftmo.get("trading_blocked", False)
            if blocked:
                r.fail("FTMO: trading_blocked=True — drawdown limit hit")
            else:
                r.ok("FTMO: trading not blocked")

            daily_dd  = float(ftmo.get("current_daily_drawdown_pct", 0.0))
            total_dd  = float(ftmo.get("current_total_drawdown_pct", 0.0))

            if abs(daily_dd) >= 4.5:
                r.fail(f"FTMO: daily DD {daily_dd:.2f}% — at hard limit!")
            elif abs(daily_dd) >= 3.0:
                r.warn(f"FTMO: daily DD {daily_dd:.2f}% — at soft circuit-breaker")
            else:
                r.ok(f"FTMO: daily DD {daily_dd:.2f}%")

            if abs(total_dd) >= 9.0:
                r.fail(f"FTMO: total DD {total_dd:.2f}% — elevated")
            else:
                r.ok(f"FTMO: total DD {total_dd:.2f}%")

        except Exception as e:
            r.warn(f"ftmo_state.json read error: {e}")
    else:
        r.warn("ftmo_state.json not found — FTMO state will initialise fresh on startup")

    return r


def check_stale_locks(verbose: bool) -> CheckResult:
    r = CheckResult("Instance Locks & Duplicate Prevention")

    if LOCK_FILE.exists():
        try:
            content = LOCK_FILE.read_text().strip()
            age_s   = time.time() - LOCK_FILE.stat().st_mtime
            age_h   = age_s / 3600
            if age_s < STALE_LOCK_SECONDS:
                r.fail(
                    f"Active lock file ({age_h:.1f}h old) — another instance may be running! "
                    f"Content: {content[:60]}"
                )
            else:
                r.warn(f"Stale lock file ({age_h:.1f}h old) — safe to delete: {LOCK_FILE.name}")
        except Exception as e:
            r.warn(f"Could not read lock file: {e}")
    else:
        r.ok("No instance lock — clean start confirmed")

    if platform.system() == "Windows":
        try:
            import subprocess
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq terminal64.exe", "/FO", "CSV"],
                capture_output=True, text=True, timeout=5
            ).stdout
            if "terminal64.exe" in out:
                r.ok("MT5 terminal64.exe process running")
            else:
                r.warn("MT5 terminal64.exe not detected — ensure MT5 is open before launch")
        except Exception:
            r.warn("Could not query MT5 process (non-critical)")

    return r


def check_mt5_connection(verbose: bool) -> CheckResult:
    r = CheckResult("MetaTrader 5 Connection")

    try:
        import MetaTrader5 as mt5
    except ImportError:
        r.fail("MetaTrader5 package not installed")
        return r

    if not mt5.initialize():
        r.fail(f"MT5 initialize() failed: {mt5.last_error()}")
        r.fail("Ensure MT5 terminal is open and logged into FTMO account")
        return r

    r.ok("MT5 initialize() succeeded")

    acc = mt5.account_info()
    if acc is None:
        r.fail(f"Cannot get account info: {mt5.last_error()}")
    else:
        r.ok(f"Account: {acc.login} | {acc.server}")
        r.ok(f"Balance: ${acc.balance:,.2f} | Equity: ${acc.equity:,.2f}")
        if acc.trade_allowed:
            r.ok("Trade allowed: True")
        else:
            r.fail("Trade allowed: False")

    # Check all live pairs + re-enable hidden ones
    hidden_count, ok_count, missing = 0, 0, []
    
    # Load mapping from config/ticker_mapping.csv if available
    mapping = {}
    mapping_file = BASE_DIR / "config" / "ticker_mapping.csv"
    if mapping_file.exists():
        try:
            import pandas as pd
            df_map = pd.read_csv(mapping_file)
            mapping = dict(zip(df_map['Ticker'], df_map['FTMO_Symbol']))
        except Exception as e:
            r.warn(f"Failed to load ticker_mapping.csv: {e}")

    for pair in LIVE_PAIRS:
        broker_symbol = mapping.get(pair, pair)
        info = mt5.symbol_info(broker_symbol)
        if info is None:
            missing.append(broker_symbol)
        elif not info.visible:
            mt5.symbol_select(broker_symbol, True)
            hidden_count += 1
            r.warn(f"Symbol {broker_symbol} was hidden — re-enabled in Market Watch")
        else:
            ok_count += 1
            if verbose:
                r.ok(f"{broker_symbol}: spread={info.spread} pts, visible=True")

    if missing:
        for s in missing:
            r.fail(f"Symbol not found in MT5: {s}")
    else:
        r.ok(f"{ok_count} symbols visible, {hidden_count} re-enabled")

    # Market hours check
    now = datetime.datetime.now(datetime.timezone.utc)
    if now.weekday() >= 5:
        r.warn(f"Today is {now.strftime('%A')} UTC — markets closed; connection check only")
    else:
        r.ok(f"Market session: UTC {now.strftime('%H:%M %A')}")

    mt5.shutdown()
    r.ok("MT5 shutdown cleanly")
    return r


def check_validation_results(verbose: bool) -> CheckResult:
    r = CheckResult("v10.5.0 Validation Results")

    result_files = sorted(RESULTS_DIR.glob("validation_v10_5_0_*.json"), reverse=True)
    if not result_files:
        r.fail(
            "No v10.5.0 validation results found.\n"
            "  Run: python CORE_MODULES/validation/validate_v10_5_0.py"
        )
        return r

    latest = result_files[0]
    mtime  = datetime.datetime.fromtimestamp(latest.stat().st_mtime)
    age_h  = (datetime.datetime.now() - mtime).total_seconds() / 3600

    if age_h > 72:
        r.warn(f"Validation results are {age_h:.0f}h old — consider re-running")
    else:
        r.ok(f"Latest results: {latest.name} ({age_h:.1f}h old)")

    try:
        with open(latest) as f:
            results = json.load(f)

        agg = results.get("aggregate", {})
        wf_wr = agg.get("win_rate", 0.0)
        wf_pf = agg.get("profit_factor", 0.0)
        pairs_validated = agg.get("pairs_validated", agg.get("valid_pairs", 0))
        total_trades    = agg.get("total_trades", 0)

        r.ok(f"Pairs validated: {pairs_validated}")
        r.ok(f"Total out-of-sample trades: {total_trades:,}")
        r.ok(f"Walk-forward WR: {wf_wr:.1%} | PF: {wf_pf:.2f}")

        # Derive verdict from PF (matches validate_v10_5_0.py exit logic)
        if wf_pf >= 1.70:
            r.ok(f"Verdict: GO (PF {wf_pf:.2f} ≥ 1.70)")
        elif wf_pf >= 1.00:
            r.warn(f"Verdict: CONDITIONAL GO (PF {wf_pf:.2f} — below 1.70 target)")
        else:
            r.fail(f"Verdict: NO-GO (PF {wf_pf:.2f} < 1.00)")

        # Baseline comparison
        cmp = results.get("comparison", {})
        pf_delta = cmp.get("profit_factor_delta", 0.0)
        wr_delta = cmp.get("win_rate_delta", 0.0)
        if pf_delta != 0 or wr_delta != 0:
            r.ok(f"vs v9.9.11 baseline: PF Δ{pf_delta:+.2f} | WR Δ{wr_delta:+.1%}")

        # Flag pairs with degraded status
        for pr in results.get("pair_tf_results", []):
            if pr and pr.get("status") == "DEGRADED":
                r.warn(f"Degraded: {pr.get('pair','?')}_{pr.get('tf','?')}")

    except Exception as e:
        r.warn(f"Could not parse validation results: {e}")

    return r


def check_codebase_integrity(verbose: bool) -> CheckResult:
    r = CheckResult("Codebase Integrity")

    for name, path in REQUIRED_SCRIPTS.items():
        if path.exists():
            r.ok(f"{name} at {path.relative_to(BASE_DIR)}")
        else:
            r.warn(f"{name} not found at expected path ({path.relative_to(BASE_DIR)})")

    # Root misplacement check — key modules should NOT be at project root
    suspects = [
        "emergency_position_sizing.py",
        "smc_confluence.py",
        "signal_fusion.py",
        "risk_governor.py",
        "position_sizer.py",
    ]
    for s in suspects:
        if (BASE_DIR / s).exists():
            r.warn(f"Root misplacement: {s} at project root — should be in CORE_MODULES/")
        else:
            if verbose:
                r.ok(f"{s} not at root (correct)")

    # Validation scripts
    for vs in ["validate_v10_5_0.py", "pre_monday_checklist.py"]:
        if (VALIDATION_DIR / vs).exists():
            r.ok(f"Validation script: {vs}")
        else:
            r.fail(f"Validation script missing: {vs}")

    return r


def check_learning_loop(verbose: bool) -> CheckResult:
    r = CheckResult("Learning Loop & Retraining Status")

    candidates = [
        BASE_DIR / "learning_loop_state.json",
        CORE_DIR / "learning_loop_state.json",
        CORE_DIR / "training" / "learning_loop_state.json",
        CONFIG_DIR / "learning_loop_state.json",
    ]
    state_file = next((p for p in candidates if p.exists()), None)

    if state_file:
        try:
            with open(state_file) as f:
                ls = json.load(f)
            r.ok(f"Learning loop state: {state_file.name}")
            last = ls.get("last_retrain_timestamp")
            if last:
                ts = datetime.datetime.fromisoformat(last)
                age_d = (datetime.datetime.now() - ts).days
                if age_d > 14:
                    r.warn(f"Last retrain: {age_d}d ago — consider retraining")
                else:
                    r.ok(f"Last retrain: {age_d}d ago")
            if ls.get("force_retrain", False):
                r.warn("force_retrain=True — models will retrain on startup (~53s)")
            else:
                r.ok("force_retrain=False — cache-first startup (~0.1s)")
        except Exception as e:
            r.warn(f"Could not parse learning loop state: {e}")
    else:
        r.warn("Learning loop state file not found — will use cached models on startup")

    # Model count
    shards = (
        list((MODEL_DIR / "models_all").rglob("lgb_*.txt"))
        + list((MODEL_DIR / "models_all").rglob("xgb_*.json"))
        + list((MODEL_DIR / "models_all").rglob("cat_*.cbm"))
    )
    if shards:
        r.ok(f"{len(shards):,} MLS V3 model artifacts across {MODEL_DIR.name}/")
    else:
        r.warn("No MLS V3 model artifacts found in models_live/models_all/")

    return r


# ===========================================================================
# REPORT & MAIN
# ===========================================================================

def print_header():
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  CAVALIER BASTION V4.0 — PRE-LIVE-TRADING READINESS CHECK{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")
    print(f"  Timestamp : {ts}")
    print(f"  ROOT      : {BASE_DIR}")
    print(f"  Live pairs: {len(LIVE_PAIRS)} active | {len(QUARANTINED_PAIRS)} quarantined")
    print(f"  Platform  : {platform.system()} {platform.release()}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"{'='*65}\n")


def print_footer(sections: List[CheckResult], skip_mt5: bool) -> int:
    total_pass = sum(1 for s in sections if s.status == "PASS")
    total_warn = sum(1 for s in sections if s.status == "WARN")
    total_fail = sum(1 for s in sections if s.status == "FAIL")

    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"{'='*65}")
    print(f"  {GREEN}PASS{RESET}: {total_pass} | {YELLOW}WARN{RESET}: {total_warn} | {RED}FAIL{RESET}: {total_fail}")

    if skip_mt5:
        print(f"  {YELLOW}Note:{RESET} MT5 connection check was skipped (--skip-mt5)")

    print()
    if total_fail == 0 and total_warn == 0:
        print(f"  {GREEN}{BOLD}✓  ALL CLEAR — SAFE TO START LIVE TRADING{RESET}")
        verdict, exit_code = "ALL_CLEAR", 0
    elif total_fail == 0:
        print(f"  {YELLOW}{BOLD}⚠  WARNINGS — REVIEW BEFORE TRADING{RESET}")
        verdict, exit_code = "WARNINGS", 1
    else:
        print(f"  {RED}{BOLD}✗  CRITICAL FAILURES — DO NOT START LIVE TRADING{RESET}")
        print(f"  {RED}   Resolve all FAIL items above before launching.{RESET}")
        verdict, exit_code = "BLOCKED", 2

    print(f"\n  Verdict: {verdict}")
    print(f"{'='*65}\n")

    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = RESULTS_DIR / f"pre_monday_checklist_{ts_str}.json"
        payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "verdict": verdict,
            "exit_code": exit_code,
            "root": str(BASE_DIR),
            "sections": [
                {
                    "name": s.name,
                    "status": s.status,
                    "failures": s.failures,
                    "warnings": s.warnings,
                    "passed_count": len(s.passed),
                }
                for s in sections
            ],
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Checklist saved: {out.name}\n")
    except Exception as e:
        print(f"  {YELLOW}⚠{RESET}  Could not save checklist: {e}\n")

    return exit_code


def main():
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    parser = argparse.ArgumentParser(description="Cavalier V4.0 — Pre-Live Readiness Check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all passed checks")
    parser.add_argument("--skip-mt5",       action="store_true", help="Skip MT5 check (offline use)")
    args = parser.parse_args()

    print_header()

    sections: List[CheckResult] = []

    for fn in [
        check_environment,
        check_disk_space,
        check_data_files,
        check_models,
        check_risk_config,
        check_stale_locks,
        check_codebase_integrity,
        check_validation_results,
        check_learning_loop,
    ]:
        result = fn(args.verbose)
        result.print_section(verbose=args.verbose)
        sections.append(result)

    if not args.skip_mt5:
        mt5_result = check_mt5_connection(args.verbose)
        mt5_result.print_section(verbose=args.verbose)
        sections.append(mt5_result)

    exit_code = print_footer(sections, skip_mt5=args.skip_mt5)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
