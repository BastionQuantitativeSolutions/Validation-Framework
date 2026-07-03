"""
# Author: JG
FAST VALIDATION - EURUSD H1 Only
Quick verification of model performance
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "DATA_MODELS" / "training"))

import pandas as pd
import numpy as np
import json
import joblib
from compute_features_ultimate import compute_all_features
from train_ml_final import create_target, get_params, force_balance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

PARQUET_DIR = Path("./sample_project/DATA_MODELS/data_parquet")
MODELS_DIR = Path("./sample_project/DATA_MODELS/models_live")
RESULTS_DIR = Path("./sample_project/CORE_MODULES/validation/results")
RESULTS_DIR.mkdir(exist_ok=True)

PAIR = "EURUSD"
TF = "H1"


def walk_forward_validate(X, y, model_class, params, n_splits=5):
    """Walk-forward validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    test_accs = []
    train_accs = []
    aucs = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train_bal, y_train_bal = force_balance(X_train, y_train)

        model = model_class(**params)
        if "cat" in str(type(model)).lower():
            model.fit(X_train_bal, y_train_bal, verbose=False)
        else:
            model.fit(X_train_bal, y_train_bal)

        y_pred = model.predict(X_test)

        test_accs.append(accuracy_score(y_test, y_pred))
        train_accs.append(accuracy_score(y_train_bal, model.predict(X_train_bal)))

        try:
            if hasattr(model, "predict_proba"):
                aucs.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
            else:
                aucs.append(roc_auc_score(y_test, y_pred))
        except Exception:
            aucs.append(0.5)

    return {
        "test_accuracy": np.mean(test_accs),
        "train_accuracy": np.mean(train_accs),
        "gap": np.mean(train_accs) - np.mean(test_accs),
        "auc": np.mean(aucs),
    }


def monte_carlo_test(X, y, model_class, params, n_iterations=1000):
    """Monte Carlo with 1k simulations"""
    np.random.seed(42)

    accuracies = []
    aucs = []
    biases = []

    for i in range(n_iterations):
        if (i + 1) % 200 == 0:
            print(f"    MC Progress: {i + 1}/{n_iterations}")

        n_samples = int(len(X) * 0.7)
        start = np.random.randint(0, len(X) - n_samples)
        window = slice(start, start + n_samples)

        X_sample = X.iloc[window].copy()
        y_sample = y.iloc[window].copy()

        split = int(len(X_sample) * 0.7)
        X_train, X_test = X_sample.iloc[:split], X_sample.iloc[split:]
        y_train, y_test = y_sample.iloc[:split], y_sample.iloc[split:]

        X_train_bal, y_train_bal = force_balance(X_train, y_train)

        if len(X_train_bal) < 10 or len(X_test) < 10:
            continue

        model = model_class(**params)
        if "cat" in str(type(model)).lower():
            model.fit(X_train_bal, y_train_bal, verbose=False)
        else:
            model.fit(X_train_bal, y_train_bal)

        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        biases.append(y_pred.mean())

        try:
            if hasattr(model, "predict_proba"):
                aucs.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
            else:
                aucs.append(roc_auc_score(y_test, y_pred))
        except Exception:
            aucs.append(0.5)

    return {
        "accuracy": np.mean(accuracies),
        "std": np.std(accuracies),
        "auc": np.mean(aucs),
        "bias": np.mean(biases),
    }


def main():
    print("=" * 60)
    print(f"FAST VALIDATION: {PAIR} {TF}")
    print("=" * 60)

    pair_tf_dir = MODELS_DIR / f"{PAIR}_{TF}"
    if not pair_tf_dir.exists():
        print(f"ERROR: Model dir not found: {pair_tf_dir}")
        return

    df = pd.read_parquet(PARQUET_DIR / f"{PAIR}_{TF}.parquet")
    print(f"Data: {len(df)} rows")

    y = create_target(df)
    X = compute_all_features(df)

    combined = pd.concat([X, y.rename("target")], axis=1).dropna()
    X_aligned = combined.drop("target", axis=1).reset_index(drop=True)
    y_aligned = combined["target"].reset_index(drop=True)
    print(f"Features: {X_aligned.shape[1]}, Samples: {len(X_aligned)}")

    results = {}

    for tier in ["full", "core"]:
        tier_dir = pair_tf_dir / tier
        if not tier_dir.exists():
            continue

        print(f"\n  Tier: {tier}")
        results[tier] = {}

        for model_name in ["xgb", "lgb", "cat"]:
            model_file = tier_dir / f"{model_name}_model.joblib"
            if not model_file.exists():
                continue

            print(f"\n  Testing {tier}/{model_name}")

            model = joblib.load(model_file)
            params = get_params(model_name)
            if model_name == "cat":
                params["depth"] = 3
            else:
                params["max_depth"] = 3
                params["n_estimators"] = 100

            print("    Walk-forward validation...")
            wf = walk_forward_validate(X_aligned, y_aligned, type(model), params, n_splits=5)

            print("    Monte Carlo (1k simulations)...")
            mc = monte_carlo_test(X_aligned, y_aligned, type(model), params, n_iterations=1000)

            results[tier][model_name] = {
                "walk_forward": wf,
                "monte_carlo": mc,
            }

            print(f"    {model_name}: WF={wf['test_accuracy'] * 100:.1f}% | MC={mc['accuracy'] * 100:.1f}% | Gap={wf['gap'] * 100:.1f}%")

    out_file = RESULTS_DIR / f"fast_validation_{PAIR}_{TF}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
