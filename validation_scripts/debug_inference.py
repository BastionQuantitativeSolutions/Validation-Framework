import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(r"c:\Users\jack\Cavalier")))

import pandas as pd
from CORE_MODULES.core.models.loader import load_tiered_models
from CORE_MODULES.core.models.ensemble import get_tiered_prediction


def test_inference(pair="EURUSD", tf="M15"):
    print(f"Testing {pair}_{tf} inference pipeline with get_tiered_prediction...", flush=True)
    df = pd.read_parquet(rf"C:\Users\jack\Cavalier\DATA_MODELS\data_parquet\{pair}_{tf}.parquet")
    print(f"Loaded {len(df)} rows from parquet.", flush=True)

    print("Loading tiered models...")
    tiered_pack, scalers_dict, features_dict = load_tiered_models(pair, tf)

    if not tiered_pack:
        print("Failed to load tiered models. Aborting.")
        return

    print("Running inference over the last 50 bars...")
    predictions = []

    # We want to test the last 50 bars
    # Using a 250 bar lookback window per prediction to allow all features (like MA_200) to populate correctly
    start_idx = len(df) - 50
    for i in range(start_idx, len(df)):
        window_df = df.iloc[max(0, i - 250) : i + 1].copy()

        result = get_tiered_prediction(tiered_pack, features_dict, window_df, confidence_threshold=0.65)

        prob = result.get("weighted_confidence", 0.5)
        predictions.append(prob)

        if (i - start_idx) % 10 == 0:
            print(f"Processed {i - start_idx}/50 steps... confidence: {prob:.4f}")

    print(f"Mean Prediction: {np.mean(predictions):.4f}")
    print(f"Max Prediction: {np.max(predictions):.4f}")
    print(f"Min Prediction: {np.min(predictions):.4f}")
    print(f"Number of signals > 0.55: {sum(1 for p in predictions if p > 0.55)}")
    print(f"Number of signals < 0.45: {sum(1 for p in predictions if p < 0.45)}")


if __name__ == "__main__":
    test_inference("EURUSD", "M15")
