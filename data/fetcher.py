"""Download and cache daily OHLCV data from Yahoo Finance."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = [
    "TSM", "NVDA", "AMD", "INTC", "ASML", "AVGO", "QCOM",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "SPY", "QQQ", "SOXX",
]

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.parquet"


def fetch_ticker(ticker: str, period: str = "2y", force: bool = False) -> pd.DataFrame:
    """Download full history for a ticker (default 2 years). Overwrites cache."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel("Ticker")
    df.index.name = "Date"
    df.to_parquet(_cache_path(ticker))
    return df


def load_cached(ticker: str) -> pd.DataFrame | None:
    """Return cached DataFrame or None if not cached."""
    p = _cache_path(ticker)
    if p.exists():
        return pd.read_parquet(p)
    return None


def update_ticker(ticker: str) -> pd.DataFrame:
    """Incrementally update cached data with latest bars.

    If no cache exists, does a full 2-year fetch.
    Otherwise downloads only from the last cached date onward and appends.
    """
    cached = load_cached(ticker)
    if cached is None or cached.empty:
        return fetch_ticker(ticker)

    last_date = cached.index.max()
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    today = dt.date.today().strftime("%Y-%m-%d")

    if start >= today:
        return cached  # already up to date

    new = yf.download(ticker, start=start, end=today, auto_adjust=True, progress=False)
    if new.empty:
        return cached

    if isinstance(new.columns, pd.MultiIndex):
        new.columns = new.columns.droplevel("Ticker")
    new.index.name = "Date"

    combined = pd.concat([cached, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    combined.to_parquet(_cache_path(ticker))
    return combined


def fetch_all(tickers: list[str] | None = None, force: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch or update all tickers. Returns {ticker: DataFrame}."""
    tickers = tickers or TICKERS
    results = {}
    for t in tickers:
        if force:
            results[t] = fetch_ticker(t)
        else:
            results[t] = update_ticker(t)
    return results


if __name__ == "__main__":
    print(f"Fetching {len(TICKERS)} tickers...\n")
    data = fetch_all(force=True)
    for ticker, df in data.items():
        print(f"  {ticker:5s}  shape={str(df.shape):12s}  "
              f"{df.index.min().date()} → {df.index.max().date()}")
    print(f"\nCache dir: {CACHE_DIR.resolve()}")
