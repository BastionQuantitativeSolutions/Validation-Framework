"""
TickVault Data Fetcher for Cavalier Backtesting

Downloads historical tick data using TickVault for absolute backtest parity.
Provides streaming interface compatible with live trading simulation.
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Iterator, List, Dict, Any
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Single tick data point."""

    timestamp: datetime
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class BarData:
    """OHLCV bar aggregation."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str


class TickVaultFetcher:
    """
    Downloads and caches historical tick data using TickVault.

    Features:
    - Parallel, resumable downloads
    - Local caching to avoid re-downloading
    - Streaming interface for replay engine
    - OHLCV bar aggregation on demand
    """

    def __init__(self, base_directory: Optional[Path] = None, tickvault_available: bool = True):
        self.base_directory = base_directory or Path("C:/Users/jack/Cavalier/DATA_MODELS/tick_data")
        self.tickvault_available = tickvault_available
        self._ensure_directory()

    def _ensure_directory(self):
        """Create base directory if it doesn't exist."""
        self.base_directory.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str, date: datetime) -> Path:
        """Get cache file path for symbol and date."""
        return self.base_directory / symbol / f"{symbol}_{date.strftime('%Y%m%d')}.parquet"

    def is_cached(self, symbol: str, date: datetime) -> bool:
        """Check if data is already cached."""
        return self._get_cache_path(symbol, date).exists()

    def _aggregate_to_bars(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Aggregate tick data to OHLCV bars.

        Timeframes: M1, M5, M15, M30, H1, H4, D1
        """
        tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}

        minutes = tf_minutes.get(timeframe, 5)
        df = df.copy()
        df.set_index("timestamp", inplace=True)

        bars = df.resample(f"{minutes}T").agg({"bid": "first", "ask": "last", "bid_volume": "sum", "ask_volume": "sum"}).dropna()

        bars["mid"] = (bars["bid"] + bars["ask"]) / 2
        bars["high"] = bars["mid"]
        bars["low"] = bars["mid"]
        bars["close"] = bars["mid"]
        bars["open"] = bars["mid"]
        bars["volume"] = bars["bid_volume"] + bars["ask_volume"]

        bars = bars[["open", "high", "low", "close", "volume"]]
        bars.reset_index(inplace=True)
        bars.rename(columns={"index": "timestamp"}, inplace=True)

        return bars

    async def download_range(self, symbol: str, start_date: datetime, end_date: datetime, force_redownload: bool = False) -> Dict[str, Any]:
        """
        Download tick data for a date range.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            start_date: Start of date range
            end_date: End of date range
            force_redownload: Override cached data

        Returns:
            Download statistics dictionary
        """
        if not self.tickvault_available:
            return self._download_fallback(symbol, start_date, end_date)

        try:
            from tick_vault import download_range as tv_download

            symbol_dir = self.base_directory / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)

            stats = {
                "symbol": symbol,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "dates_downloaded": [],
                "dates_cached": [],
                "dates_failed": [],
            }

            current = start_date
            while current <= end_date:
                cache_path = self._get_cache_path(symbol, current)

                if cache_path.exists() and not force_redownload:
                    stats["dates_cached"].append(current.isoformat())
                    logger.info(f"[tickvault] {symbol} {current.date()} already cached")
                else:
                    try:
                        logger.info(f"[tickvault] Downloading {symbol} {current.date()}...")

                        await tv_download(symbol=symbol.upper(), start=current, end=current + timedelta(days=1), base_directory=str(symbol_dir))
                        stats["dates_downloaded"].append(current.isoformat())
                        logger.info(f"[tickvault] {symbol} {current.date()} downloaded successfully")

                    except Exception as e:
                        stats["dates_failed"].append(current.isoformat())
                        logger.warning(f"[tickvault] {symbol} {current.date()} failed: {e}")

                current += timedelta(days=1)

            return stats

        except ImportError:
            logger.warning("[tickvault] tick-vault not installed, using fallback downloader")
            return self._download_fallback(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"[tickvault] Download error: {e}")
            return self._download_fallback(symbol, start_date, end_date)

    def _download_fallback(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Fallback download using dukascopy-python if TickVault unavailable.
        Uses dukascopy_python library with fetch() API.
        """
        try:
            import dukascopy_python as dukascopy

            stats = {
                "symbol": symbol,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "dates_downloaded": [],
                "dates_cached": [],
                "dates_failed": [],
                "method": "dukascopy_fallback",
            }

            current = start_date
            while current <= end_date:
                cache_path = self._get_cache_path(symbol, current)

                if cache_path.exists():
                    stats["dates_cached"].append(current.isoformat())
                else:
                    try:
                        logger.info(f"[dukascopy] Downloading {symbol} {current.date()}...")

                        df = dukascopy.fetch(
                            instrument=symbol.upper(),
                            start=current,
                            end=current + timedelta(days=1),
                            interval=dukascopy.INTERVAL_TICK,
                            offer_side=dukascopy.OFFER_SIDE_BID,
                        )

                        if df is not None and not df.empty:
                            df = df.rename(
                                columns={
                                    "time": "timestamp",
                                    "bid_open": "bid",
                                    "bid_high": "bid",
                                    "bid_low": "bid",
                                    "bid_close": "bid",
                                    "ask_open": "ask",
                                    "ask_high": "ask",
                                    "ask_low": "ask",
                                    "ask_close": "ask",
                                }
                            )
                            df = df[["timestamp", "bid", "ask", "bid_volume", "ask_volume"]].copy()
                            df.to_parquet(cache_path, index=False)
                            stats["dates_downloaded"].append(current.isoformat())
                            logger.info(f"[dukascopy] {symbol} {current.date()} - {len(df)} ticks saved")
                        else:
                            stats["dates_failed"].append(current.isoformat())
                            logger.warning(f"[dukascopy] {symbol} {current.date()} - no data returned")

                    except Exception as e:
                        stats["dates_failed"].append(current.isoformat())
                        logger.warning(f"[dukascopy] {symbol} {current.date()} failed: {e}")

                current += timedelta(days=1)

            return stats

        except ImportError:
            logger.error("[backtest] Neither tick-vault nor dukascopy-python installed")
            return {"error": "No tick data library available"}

    def load_ticks(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load tick data from cache into DataFrame.
        """
        all_ticks = []

        current = start_date
        while current <= end_date:
            cache_path = self._get_cache_path(symbol, current)

            if cache_path.exists():
                try:
                    df = pd.read_parquet(cache_path)
                    all_ticks.append(df)
                except Exception as e:
                    logger.warning(f"[tickvault] Failed to load {cache_path}: {e}")
            else:
                logger.warning(f"[tickvault] No cache found for {symbol} {current.date()}")

            current += timedelta(days=1)

        if all_ticks:
            combined = pd.concat(all_ticks, ignore_index=True)
            combined["timestamp"] = pd.to_datetime(combined["timestamp"])
            combined = combined.sort_values("timestamp")
            return combined
        else:
            return pd.DataFrame()

    def load_bars(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str = "M5") -> pd.DataFrame:
        """
        Load OHLCV bars from existing parquet data or aggregate from ticks.
        """
        parquet_dir = Path("C:/Users/jack/Cavalier/DATA_MODELS/data_parquet")
        tf_map = {"M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30", "H1": "H1", "H4": "H4", "D1": "D1"}
        tf_file = tf_map.get(timeframe.upper(), timeframe.upper())

        parquet_file = parquet_dir / f"{symbol.upper()}_{tf_file}.parquet"
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)

                if df.index.name == "time":
                    df = df.reset_index()

                if "timestamp" in df.columns:
                    df["time"] = pd.to_datetime(df["timestamp"])
                elif "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
                    df["time"] = pd.to_datetime(df["time"])

                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                    start = pd.Timestamp(start_date)
                    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
                    mask = (df["time"] >= start) & (df["time"] < end)
                    df = df[mask].copy()
                else:
                    return pd.DataFrame()

                if df is not None and not df.empty:
                    logger.info(f"[backtest] Loaded {len(df)} bars for {symbol} {timeframe} from parquet")
                    return df
            except Exception as e:
                logger.warning(f"[backtest] Failed to load parquet: {e}")

        ticks = self.load_ticks(symbol, start_date, end_date)

        if ticks.empty:
            return pd.DataFrame()

        return self._aggregate_to_bars(ticks, timeframe)

    def stream_ticks(self, symbol: str, start_date: datetime, end_date: datetime, speed_multiplier: float = 1.0) -> Iterator[TickData]:
        """
        Stream tick data for replay simulation.

        Args:
            symbol: Trading symbol
            start_date: Start of data range
            end_date: End of data range
            speed_multiplier: 1.0 = real-time, 10.0 = 10x speed

        Yields:
            TickData objects at simulated speed
        """
        df = self.load_ticks(symbol, start_date, end_date)

        if df.empty:
            logger.warning(f"[backtest] No tick data available for {symbol}")
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        base_time = df["timestamp"].iloc[0]

        for _, row in df.iterrows():
            tick_time = row["timestamp"]

            if speed_multiplier != 1.0:
                elapsed = (tick_time - base_time).total_seconds() / speed_multiplier
                simulated_time = base_time + timedelta(seconds=elapsed)
                yield TickData(
                    timestamp=simulated_time,
                    bid=float(row["bid"]),
                    ask=float(row["ask"]),
                    bid_volume=float(row.get("bid_volume", 0)),
                    ask_volume=float(row.get("ask_volume", 0)),
                )
            else:
                yield TickData(
                    timestamp=tick_time,
                    bid=float(row["bid"]),
                    ask=float(row["ask"]),
                    bid_volume=float(row.get("bid_volume", 0)),
                    ask_volume=float(row.get("ask_volume", 0)),
                )

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get information about cached data for a symbol."""
        symbol_dir = self.base_directory / symbol

        if not symbol_dir.exists():
            return {"cached_days": 0, "date_range": None}

        cache_files = list(symbol_dir.glob("*.parquet"))
        dates = []

        for f in cache_files:
            try:
                date_str = f.stem.split("_")[-1]
                dates.append(datetime.strptime(date_str, "%Y%m%d"))
            except ValueError:
                continue

        if dates:
            return {
                "cached_days": len(dates),
                "date_range": {"start": min(dates).isoformat(), "end": max(dates).isoformat()},
                "cache_directory": str(symbol_dir),
            }

        return {"cached_days": 0, "date_range": None}


async def download_symbols_async(
    symbols: List[str], start_date: datetime, end_date: datetime, base_directory: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Download data for multiple symbols in parallel.
    """
    fetcher = TickVaultFetcher(base_directory=base_directory)

    tasks = [fetcher.download_range(symbol, start_date, end_date) for symbol in symbols]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    summary = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, Exception):
            summary[symbol] = {"error": str(result)}
        else:
            summary[symbol] = result

    return summary


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    symbols = ["EURUSD", "GBPUSD", "USDJPY"]

    if len(sys.argv) > 1:
        if sys.argv[1] == "download":
            start = datetime(2026, 1, 1)
            end = datetime(2026, 1, 31)

            results = asyncio.run(download_symbols_async(symbols, start, end))

            for symbol, result in results.items():
                print(f"\n{symbol}:")
                print(f"  Downloaded: {len(result.get('dates_downloaded', []))} days")
                print(f"  Cached: {len(result.get('dates_cached', []))} days")
                print(f"  Failed: {len(result.get('dates_failed', []))} days")

        elif sys.argv[1] == "info":
            fetcher = TickVaultFetcher()
            for symbol in symbols:
                info = fetcher.get_symbol_info(symbol)
                print(f"\n{symbol}:")
                print(f"  {info}")
