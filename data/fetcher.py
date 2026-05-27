# ============================================================
# Smart Money Tracker
# Copyright (c) 2026 Dhruv Patel. All rights reserved.
#
# This software is proprietary and confidential.
# Unauthorized copying, distribution, or modification
# of this file, via any medium, is strictly prohibited.
#
# Author:  Dhruv Patel
# GitHub:  github.com/dhruvpatel29
# Email:   dhruvkumarp79@gmail.com
# ============================================================

"""
data/fetcher.py — Fetch OHLCV price data and macro data.

Uses yfinance (completely free, no API key needed).
Fetches: stock OHLCV, BTC price, VIX (fear index), DXY (dollar index).
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def fetch_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker using yfinance.

    Args:
        ticker: e.g. "AAPL", "BTC-USD", "NVDA"
        period: "1y", "2y", "5y", "max"

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex

    Example:
        df = fetch_ohlcv("AAPL", period="2y")
        print(df.tail())
    """
    print(f"Fetching {ticker} ({period})...")
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Standardize column names to lowercase
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.columns = [c.replace(' ', '_') for c in df.columns]
    df.index.name = "timestamp"

    # Drop rows with missing close price
    df = df.dropna(subset=["close"])

    print(f"  Fetched {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    return df


def fetch_vix() -> pd.DataFrame:
    """
    Fetch VIX (CBOE Volatility Index) — the "fear index".
    VIX > 30 = high fear, market likely volatile.
    VIX < 15 = complacency, potential for sharp moves.
    """
    return fetch_ohlcv("^VIX", period="2y")


def fetch_dxy() -> pd.DataFrame:
    """
    Fetch DXY (US Dollar Index).
    Strong dollar (rising DXY) = headwind for stocks and crypto.
    """
    return fetch_ohlcv("DX-Y.NYB", period="2y")


def fetch_sector_etf(sector: str = "XLK") -> pd.DataFrame:
    """
    Fetch a sector ETF for relative strength calculation.
    Common sector ETFs:
      XLK = Technology    XLF = Financials    XLE = Energy
      XLV = Healthcare    XLI = Industrials   XLY = Consumer
    """
    return fetch_ohlcv(sector, period="2y")


def fetch_spy() -> pd.DataFrame:
    """Fetch SPY (S&P 500 ETF) for market-wide context."""
    return fetch_ohlcv("SPY", period="2y")


def compute_sector_relative_strength(
    ticker_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    window: int = 20
) -> pd.Series:
    """
    Relative strength of a stock vs its sector ETF.
    RS = (stock return over window) / (sector return over window)
    RS > 1.0 means the stock is outperforming its sector (bullish).
    """
    stock_return  = ticker_df["close"].pct_change(window)
    sector_return = sector_df["close"].pct_change(window)

    # Align on the same dates
    stock_return, sector_return = stock_return.align(sector_return, join="inner")

    rs = stock_return / sector_return.replace(0, float("nan"))
    rs.name = "sector_rs"
    return rs


def fetch_all_data(ticker: str) -> dict:
    """
    Fetch everything needed for one ticker.
    Returns a dict of DataFrames ready for indicator calculation.

    Usage:
        data = fetch_all_data("AAPL")
        price_df = data["price"]
        vix_df   = data["vix"]
    """
    data = {}
    data["price"] = fetch_ohlcv(ticker, period="2y")
    data["vix"]   = fetch_vix()
    data["dxy"]   = fetch_dxy()
    data["spy"]   = fetch_spy()

    # Map ticker to its sector ETF
    sector_map = {
        "AAPL": "XLK", "NVDA": "XLK", "MSFT": "XLK", "AMZN": "XLY",
        "TSLA": "XLY", "BTC-USD": "XLK",  # proxy
    }
    sector = sector_map.get(ticker, "XLK")
    data["sector"] = fetch_sector_etf(sector)

    return data


if __name__ == "__main__":
    # Quick test — run: python data/fetcher.py
    df = fetch_ohlcv("AAPL", period="1y")
    print("\nAAPL last 5 rows:")
    print(df.tail())

    vix = fetch_vix()
    print(f"\nVIX latest close: {vix['close'].iloc[-1]:.2f}")
