import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PAIRS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
TF = mt5.TIMEFRAME_M5
TF_NAME = "M5_1Y"

OUTPUT_DIR = Path("./sample_project/DATA_MODELS/data_1y_backtest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_1y_data():
    if not mt5.initialize():
        logging.error("MT5 initialization failed")
        return

    # 1 Year of M5 bars = ~75,000 bars
    num_bars = 75000
    logging.info(f"Fetching 1-Year Data ({num_bars} bars)...")

    for pair in PAIRS:
        logging.info(f"Fetching {pair}...")

        rates = mt5.copy_rates_from_pos(pair, TF, 0, num_bars)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            output_path = OUTPUT_DIR / f"{pair}_{TF_NAME}.parquet"
            df.to_parquet(output_path)
            logging.info(f"  Saved {len(df)} bars to {output_path}")
        else:
            logging.warning(f"  Failed to fetch data for {pair}")

    mt5.shutdown()
    logging.info("Data fetch complete!")


if __name__ == "__main__":
    fetch_1y_data()
