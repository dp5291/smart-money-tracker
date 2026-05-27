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
data/indicators.py — Compute all 25 technical indicators.

Each indicator is explained with:
  - What it measures
  - How to interpret it
  - How it feeds into the LSTM (as a normalized feature)

Run standalone to test:
  python data/indicators.py
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Tuple


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — TREND INDICATORS
# These tell you the overall direction of the market.
# Most important for the LSTM because trend is the single
# biggest predictor of future short-term direction.
# ═══════════════════════════════════════════════════════════════

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: SMA 200, SMA 50, EMA 20, distance features, Golden Cross flag.

    THE MOST IMPORTANT: 200 SMA
    ─────────────────────────────
    The 200-day Simple Moving Average is THE line that separates
    bull markets from bear markets. Institutional investors (hedge
    funds, pension funds) use it as a primary decision rule:
      - Price > 200 SMA → bull market, bias long
      - Price < 200 SMA → bear market, bias short

    We don't just feed the raw SMA — we feed the % distance from it.
    A stock 25% above its 200 SMA is very different from one 0.5%
    above it. The model learns these nuances.

    GOLDEN CROSS / DEATH CROSS
    ─────────────────────────────
    When the 50 SMA crosses ABOVE the 200 SMA → Golden Cross
    This is one of the most reliable long-term buy signals.
    When the 50 SMA crosses BELOW → Death Cross (sell signal).
    The model gets this as a binary flag.
    """
    df = df.copy()

    # Raw moving averages
    df["sma_200"] = ta.sma(df["close"], length=200)
    df["sma_50"]  = ta.sma(df["close"], length=50)
    df["ema_20"]  = ta.ema(df["close"], length=20)

    # % distance from each MA (how overextended is price?)
    # Formula: (close - MA) / MA * 100
    df["sma_200_dist"] = (df["close"] - df["sma_200"]) / df["sma_200"] * 100
    df["sma_50_dist"]  = (df["close"] - df["sma_50"])  / df["sma_50"]  * 100
    df["ema_20_dist"]  = (df["close"] - df["ema_20"])  / df["ema_20"]  * 100

    # Golden Cross flag: 1 if 50 SMA is above 200 SMA, 0 if not
    # This captures the long-term market regime
    df["golden_cross"] = (df["sma_50"] > df["sma_200"]).astype(float)

    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — MOMENTUM INDICATORS
# These measure HOW FAST price is moving and whether
# momentum is increasing or exhausting.
# ═══════════════════════════════════════════════════════════════

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: RSI, MACD, Stochastic, Rate of Change.

    RSI (Relative Strength Index)
    ─────────────────────────────
    Measures the speed and magnitude of recent price changes.
    Scale: 0 to 100.
      - RSI > 70 = overbought (likely to pull back)
      - RSI < 30 = oversold (likely to bounce)
      - RSI crossing 50 upward = bullish momentum
    One of the most used indicators by retail and institutional traders.
    We normalize it to 0-1 range for the model.

    MACD (Moving Average Convergence Divergence)
    ─────────────────────────────────────────────
    Difference between 12-day EMA and 26-day EMA.
    When MACD crosses above Signal line = bullish momentum shift.
    When histogram is growing = trend strengthening.
    The model gets 3 MACD features: line, signal, and histogram.

    STOCHASTIC OSCILLATOR
    ─────────────────────
    Where is today's close relative to the recent high-low range?
    Above 80 = overbought, Below 20 = oversold.
    Works well in ranging markets. Complements RSI.
    """
    df = df.copy()

    # RSI normalized to 0-1 (from 0-100)
    rsi_raw = ta.rsi(df["close"], length=14)
    df["rsi_14"] = rsi_raw / 100.0

    # MACD: returns DataFrame with MACD, Signal, Histogram columns
    macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        df["macd"]        = macd_df.iloc[:, 0]  # MACD line
        df["macd_signal"] = macd_df.iloc[:, 2]  # Signal line
        df["macd_hist"]   = macd_df.iloc[:, 1]  # Histogram
    else:
        df["macd"] = df["macd_signal"] = df["macd_hist"] = 0.0

    # Stochastic: %K and %D lines (both normalized to 0-1)
    stoch_df = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    if stoch_df is not None and not stoch_df.empty:
        df["stoch_k"] = stoch_df.iloc[:, 0] / 100.0
        df["stoch_d"] = stoch_df.iloc[:, 1] / 100.0
    else:
        df["stoch_k"] = df["stoch_d"] = 0.5

    # Rate of Change: % change over 10 days
    # Positive = momentum up. Negative = momentum down.
    df["roc_10"] = ta.roc(df["close"], length=10) / 100.0

    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — VOLATILITY INDICATORS
# These measure how much price is swinging.
# High volatility = more risk and bigger moves both ways.
# ═══════════════════════════════════════════════════════════════

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: Bollinger Bands (upper, lower, width, distances), ATR.

    BOLLINGER BANDS
    ───────────────
    Three lines:
      - Middle band: 20-day SMA
      - Upper band: middle + 2 standard deviations
      - Lower band: middle - 2 standard deviations

    Key setups:
      - Price touching upper band = overbought
      - Price touching lower band = oversold
      - Bollinger Squeeze (bands very narrow) = explosive move coming
        After a squeeze, the breakout direction is the new trend.
    We feed: distance to upper band, distance to lower band, band width.

    ATR (Average True Range)
    ────────────────────────
    Average size of each daily candle over 14 days.
    High ATR = volatile (use wider stops).
    Low ATR = quiet market (tight stops OK).
    We normalize by price: ATR / close (so it's scale-independent).
    """
    df = df.copy()

    # Bollinger Bands
    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is not None and not bb.empty:
        df["bb_upper"]  = bb.iloc[:, 0]   # Upper band
        df["bb_middle"] = bb.iloc[:, 1]   # Middle band (SMA 20)
        df["bb_lower"]  = bb.iloc[:, 2]   # Lower band

        # % distance from price to each band
        df["bb_upper_dist"] = (df["bb_upper"]  - df["close"]) / df["close"] * 100
        df["bb_lower_dist"] = (df["close"] - df["bb_lower"])  / df["close"] * 100

        # Band width as % of middle band (squeeze detection)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100
    else:
        df["bb_upper"] = df["bb_middle"] = df["bb_lower"] = df["close"]
        df["bb_upper_dist"] = df["bb_lower_dist"] = df["bb_width"] = 0.0

    # ATR normalized by close price
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_14"]  = atr
    df["atr_norm"] = atr / df["close"]   # Makes it price-independent

    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — VOLUME INDICATORS
# Volume is the fuel of price movement.
# High volume confirms a move. Low volume = weak, likely to fail.
# ═══════════════════════════════════════════════════════════════

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: VWAP, OBV, volume ratio, volume trend.

    VWAP (Volume Weighted Average Price)
    ──────────────────────────────────────
    The average price weighted by how much was traded at each price.
    Institutional traders use VWAP as a benchmark:
      - Price > VWAP = market is bullish intraday
      - Price < VWAP = market is bearish intraday
    Big funds try to buy below VWAP and sell above it.
    We approximate with a rolling VWAP since we use daily data.

    OBV (On Balance Volume)
    ───────────────────────
    Cumulative volume: +volume on up days, -volume on down days.
    OBV DIVERGENCE is a powerful signal:
      - Price flat or down, OBV rising = accumulation
        (smart money buying quietly → price will follow up)
      - Price rising, OBV falling = distribution
        (smart money selling into the rally → price will follow down)

    VOLUME RATIO
    ─────────────
    Today's volume vs 20-day average.
    3x average on an up day = very strong conviction.
    3x average on a down day = panic selling / distribution.
    """
    df = df.copy()

    # VWAP (rolling approximation for daily data)
    # True VWAP resets daily (intraday), so we use a 20-day rolling version
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    rolling_pv    = (typical_price * df["volume"]).rolling(20).sum()
    rolling_vol   = df["volume"].rolling(20).sum()
    df["vwap"]      = rolling_pv / rolling_vol
    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["vwap"] * 100

    # OBV
    obv = ta.obv(df["close"], df["volume"])
    df["obv"] = obv
    # Normalize OBV by dividing by its 20-day rolling std (makes it comparable)
    df["obv_norm"] = obv / (obv.rolling(20).std().replace(0, 1))

    # Volume ratio: today's volume vs 20-day average
    vol_avg = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / vol_avg.replace(0, 1)

    # Volume trend: slope of volume over last 5 days
    # Positive = volume increasing (trend gaining strength)
    # Negative = volume decreasing (trend losing conviction)
    df["volume_trend"] = df["volume"].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean()
        if x.mean() != 0 else 0,
        raw=True
    )

    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — STRUCTURE INDICATORS
# Support, resistance, and Fibonacci levels are the
# "hidden" price levels where market participants cluster.
# Professional traders watch these extremely carefully.
# ═══════════════════════════════════════════════════════════════

def add_structure_indicators(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Adds: support level, resistance level, Fibonacci 61.8%.

    SUPPORT & RESISTANCE (Auto-detected)
    ──────────────────────────────────────
    Support = price level where buyers historically stepped in
              (local minimum in a rolling window)
    Resistance = price level where sellers historically appeared
                 (local maximum in a rolling window)

    When price is near support = lower downside risk, good buy zone
    When price is near resistance = ceiling is close, risk of reversal

    We detect these automatically using rolling min/max over
    a configurable window. Then we compute % distance from current
    price to each level — this is what the model learns from.

    FIBONACCI RETRACEMENT (61.8% level)
    ─────────────────────────────────────
    The 61.8% Fibonacci retracement is the "golden ratio" level.
    After a significant move up, traders look for price to retrace
    to the 61.8% Fib level before continuing upward.
    This is one of the most watched price levels by technical traders.
    """
    df = df.copy()

    # Auto-detect support and resistance using rolling windows
    df["support"]    = df["low"].rolling(window, center=True).min()
    df["resistance"] = df["high"].rolling(window, center=True).max()

    # % distance from current price to support/resistance
    df["support_dist"]    = (df["close"] - df["support"])    / df["close"] * 100
    df["resistance_dist"] = (df["resistance"] - df["close"]) / df["close"] * 100

    # Fibonacci 61.8% retracement level
    # Calculated from the most recent swing high and swing low
    swing_high = df["high"].rolling(window * 2).max()
    swing_low  = df["low"].rolling(window * 2).min()
    df["fib_618"]      = swing_high - 0.618 * (swing_high - swing_low)
    df["fib_618_dist"] = (df["close"] - df["fib_618"]) / df["close"] * 100

    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — FEATURE NORMALIZATION
# The LSTM needs all features on similar scales.
# Without normalization, a 200 SMA value of $185 would
# completely dominate an RSI value of 0.65.
# ═══════════════════════════════════════════════════════════════

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features for LSTM input.
    - Price-based features: normalize by rolling window (relative change)
    - Ratio features: already 0-1 or small range
    - Distance features: clip to ±20% range, divide by 20
    """
    df = df.copy()

    # Normalize close and volume as % change (makes it stationary)
    df["close_norm"]  = df["close"].pct_change().fillna(0)
    df["volume_norm"] = df["volume"].pct_change().fillna(0).clip(-3, 3)

    # Clip distance features to ±20% and normalize to ±1
    dist_cols = [c for c in df.columns if c.endswith("_dist")]
    for col in dist_cols:
        df[col] = df[col].clip(-20, 20) / 20.0

    # Clip MACD-related features
    for col in ["macd", "macd_signal", "macd_hist"]:
        if col in df.columns:
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] / std).clip(-3, 3)

    # Clip OBV norm
    if "obv_norm" in df.columns:
        df["obv_norm"] = df["obv_norm"].clip(-3, 3)

    # Clip volume ratio (typical range 0-5, normalize around 1)
    if "volume_ratio" in df.columns:
        df["volume_ratio"] = (df["volume_ratio"] - 1).clip(-2, 4) / 4.0

    # ROC already as decimal, clip extremes
    if "roc_10" in df.columns:
        df["roc_10"] = df["roc_10"].clip(-0.3, 0.3) / 0.3

    # Volume trend: clip and normalize
    if "volume_trend" in df.columns:
        df["volume_trend"] = df["volume_trend"].clip(-2, 2) / 2.0

    # ATR norm: typical range 0-0.05 (0-5%), normalize to 0-1
    if "atr_norm" in df.columns:
        df["atr_norm"] = df["atr_norm"].clip(0, 0.10) / 0.10

    return df


# ═══════════════════════════════════════════════════════════════
# MASTER FUNCTION — run everything in order
# ═══════════════════════════════════════════════════════════════

def compute_all_indicators(
    df: pd.DataFrame,
    finbert_scores: pd.Series = None
) -> pd.DataFrame:
    """
    Run all indicator functions in the correct order.

    Args:
        df:              OHLCV DataFrame from fetcher.py
        finbert_scores:  Optional Series of FinBERT scores indexed by date

    Returns:
        DataFrame with all 25 features, normalized, ready for LSTM.
        NaN rows at the start (from indicator warm-up) are dropped.

    Usage:
        from data.fetcher import fetch_ohlcv
        from data.indicators import compute_all_indicators

        df = fetch_ohlcv("AAPL", period="2y")
        features = compute_all_indicators(df)
        print(features.shape)   # (N, 25+)
        print(features.tail())
    """
    print("Computing trend indicators...")
    df = add_trend_indicators(df)

    print("Computing momentum indicators...")
    df = add_momentum_indicators(df)

    print("Computing volatility indicators...")
    df = add_volatility_indicators(df)

    print("Computing volume indicators...")
    df = add_volume_indicators(df)

    print("Computing structure indicators...")
    df = add_structure_indicators(df)

    # Add FinBERT sentiment scores if provided
    if finbert_scores is not None:
        df["finbert_score"] = df.index.map(
            lambda d: finbert_scores.get(d.date(), 0.0)
        )
    else:
        df["finbert_score"] = 0.0

    print("Normalizing features...")
    df = normalize_features(df)

    # Drop rows where indicators haven't warmed up yet
    # 200 SMA needs 200 days, so first 200 rows will be NaN
    df = df.dropna(subset=["sma_200", "rsi_14", "macd"])

    print(f"Final dataset: {len(df)} rows with all indicators computed")
    return df


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list
) -> pd.DataFrame:
    """
    Extract only the columns that go into the LSTM.
    Fills any remaining NaN with 0 (shouldn't happen after compute_all_indicators).
    """
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Warning: missing features: {missing}")
        for col in missing:
            df[col] = 0.0
    return df[feature_cols].fillna(0.0)


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from data.fetcher import fetch_ohlcv

    print("="*50)
    print("Testing indicator computation on AAPL")
    print("="*50)

    df = fetch_ohlcv("AAPL", period="2y")
    features = compute_all_indicators(df)

    print("\nFeature columns:", list(features.columns))
    print(f"\nShape: {features.shape}")
    print("\nLast row (most recent trading day):")
    print(features.tail(1).T.to_string())

    # Show the key signals for today
    latest = features.iloc[-1]
    print(f"\n{'='*40}")
    print(f"AAPL Signal Summary (latest day)")
    print(f"{'='*40}")
    print(f"200 SMA distance : {latest.get('sma_200_dist', 0):.2f}% from price")
    print(f"Golden Cross     : {'YES' if latest.get('golden_cross', 0) == 1 else 'NO'}")
    print(f"RSI              : {latest.get('rsi_14', 0)*100:.1f}")
    print(f"BB Width         : {latest.get('bb_width', 0):.2f}% (squeeze if < 5%)")
    print(f"Volume ratio     : {(latest.get('volume_ratio', 0)*4+1):.2f}x avg")
    print(f"Support distance : {latest.get('support_dist', 0)*20:.2f}% above support")
