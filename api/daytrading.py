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
api/daytrading.py — Day trading endpoints for options traders.

Provides:
  - Pre-market high/low levels (auto-detected)
  - Market structure (higher highs/lows detection)
  - Time-based chart recommendation (2m/5m/10m)
  - Hammer/doji candle detection at key levels
  - Options signal (calls/puts with confidence)
  - Intraday VWAP position
"""

import os
import sys
from datetime import datetime, time, timedelta
from typing import Optional
import pytz

from fastapi import APIRouter, Depends, Request, HTTPException
import yfinance as yf
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api.security import verify_api_key, validate_ticker, limiter, RATE_LIMITS

router = APIRouter(prefix="/daytrading", tags=["Day Trading"])

ET = pytz.timezone("America/New_York")


# ── Time-based chart recommendation ───────────────────────────

def get_recommended_timeframe() -> dict:
    """
    Returns the recommended chart timeframe based on current ET time.
    Matches the trader's system:
      9:30–10:00 → 2m (+ 5m confluence)
      10:00–11:00 → 5m
      11:00+ → 10m (+ 5m confluence)
      Pre-market → 1H for level drawing
    """
    now_et = datetime.now(ET).time()

    pre_market_start = time(4, 0)
    market_open      = time(9, 30)
    phase1_end       = time(10, 0)
    phase2_end       = time(11, 0)
    market_close     = time(16, 0)

    if pre_market_start <= now_et < market_open:
        return {
            "primary":    "1H",
            "confluence": None,
            "phase":      "pre-market",
            "action":     "Draw levels on 1H. Mark pre-market high and low. No trades yet.",
            "color":      "#185FA5",
        }
    elif market_open <= now_et < phase1_end:
        return {
            "primary":    "2m",
            "confluence": "5m",
            "phase":      "market-open",
            "action":     "Watch 2m for higher highs/lows. Confirm on 5m. Look for hammer candles at levels.",
            "color":      "#22c55e",
        }
    elif phase1_end <= now_et < phase2_end:
        return {
            "primary":    "5m",
            "confluence": None,
            "phase":      "mid-morning",
            "action":     "5m structure continuation. Wait for pullbacks to key levels before entering.",
            "color":      "#f59e0b",
        }
    elif phase2_end <= now_et < market_close:
        return {
            "primary":    "10m",
            "confluence": "5m",
            "phase":      "afternoon",
            "action":     "10m for bigger moves. Use 5m to confirm entries. Wait for trend line breaks.",
            "color":      "#6366f1",
        }
    else:
        return {
            "primary":    "1H",
            "confluence": "4H",
            "phase":      "after-hours",
            "action":     "Market closed. Review trades. Plan tomorrow's levels on 1H and 4H.",
            "color":      "#6b7280",
        }


# ── Fetch intraday data ────────────────────────────────────────

def fetch_intraday(ticker: str, interval: str = "2m", period: str = "1d") -> pd.DataFrame:
    """Fetch intraday OHLCV data using yfinance."""
    try:
        df = yf.download(ticker, interval=interval, period=period,
                        progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Intraday fetch error: {e}")
        return pd.DataFrame()


# ── Pre-market levels ──────────────────────────────────────────

def get_premarket_levels(ticker: str) -> dict:
    """
    Auto-detect pre-market high and low.
    These are the most important levels of the day for options traders.
    Pre-market high = first resistance.
    Pre-market low  = first support.
    """
    try:
        # Fetch 30-minute data for today + pre-market
        df = yf.download(ticker, interval="30m", period="2d",
                        progress=False, auto_adjust=True, prepost=True)

        if df.empty:
            return {"high": None, "low": None, "error": "No data"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        # Filter to today's pre-market (4am - 9:30am ET)
        now_et = datetime.now(ET)
        today  = now_et.date()

        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(ET)

        pre_market = df[
            (df.index.date == today) &
            (df.index.time >= time(4, 0)) &
            (df.index.time < time(9, 30))
        ]

        if pre_market.empty:
            # Fallback: use yesterday's after-hours or last known levels
            last_day = df.tail(10)
            return {
                "high":   round(float(last_day["high"].max()), 2),
                "low":    round(float(last_day["low"].min()),  2),
                "source": "previous-session",
                "note":   "Pre-market not started yet — showing recent levels",
            }

        pm_high = float(pre_market["high"].max())
        pm_low  = float(pre_market["low"].min())

        return {
            "high":   round(pm_high, 2),
            "low":    round(pm_low,  2),
            "source": "today-premarket",
            "note":   "Live pre-market levels",
        }

    except Exception as e:
        return {"high": None, "low": None, "error": str(e)}


# ── Market structure detection ─────────────────────────────────

def detect_structure(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Detect higher highs / higher lows (uptrend)
    or lower highs / lower lows (downtrend).

    This is the core of the trader's entry system:
    - Uptrend (HH + HL) → bias for calls
    - Downtrend (LH + LL) → bias for puts
    - Choppy → wait, no trade
    """
    if df.empty or len(df) < lookback:
        return {"trend": "unknown", "structure": "insufficient data", "bias": "neutral", "action": "Not enough data"}

    recent = df.tail(lookback)
    highs  = recent["high"].values
    lows   = recent["low"].values

    # Find swing highs and lows (local extremes)
    swing_highs = []
    swing_lows  = []
    window = 3

    for i in range(window, len(highs) - window):
        if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, window+1)):
            swing_highs.append(highs[i])

        if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, window+1)):
            swing_lows.append(lows[i])

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {
            "trend":     "choppy",
            "structure": "Not enough swing points — wait for clear trend",
            "bias":      "neutral",
            "action":    "Wait for hammer candle at key level",
        }

    # Check last 2 swing highs and lows
    hh = swing_highs[-1] > swing_highs[-2]  # higher high
    hl = swing_lows[-1]  > swing_lows[-2]   # higher low
    lh = swing_highs[-1] < swing_highs[-2]  # lower high
    ll = swing_lows[-1]  < swing_lows[-2]   # lower low

    if hh and hl:
        return {
            "trend":     "uptrend",
            "structure": "Higher Highs + Higher Lows",
            "bias":      "bullish",
            "action":    "Look for CALLS at pullback to support / key level",
            "emoji":     "📈",
        }
    elif lh and ll:
        return {
            "trend":     "downtrend",
            "structure": "Lower Highs + Lower Lows",
            "bias":      "bearish",
            "action":    "Look for PUTS at bounce to resistance / key level",
            "emoji":     "📉",
        }
    else:
        return {
            "trend":     "choppy",
            "structure": "Mixed signals — no clear direction",
            "bias":      "neutral",
            "action":    "Wait. Only trade hammer candles at key levels.",
            "emoji":     "⚠️",
        }


# ── Hammer / Doji candle detection ────────────────────────────

def detect_hammer_candle(df: pd.DataFrame) -> dict:
    """
    Detect hammer or doji candles at the last bar.
    These are high-probability reversal signals at key levels.

    Hammer:  small body, long lower wick (bullish reversal at support)
    Inverted hammer: small body, long upper wick (bearish reversal at resistance)
    Doji:    very small body relative to range (indecision = potential reversal)
    """
    if df.empty or len(df) < 2:
        return {"detected": False}

    last = df.iloc[-1]
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])

    body        = abs(c - o)
    total_range = h - l
    if total_range == 0:
        return {"detected": False}

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body_pct   = body / total_range

    # Doji: body is less than 10% of total range
    if body_pct < 0.10:
        return {
            "detected": True,
            "type":     "doji",
            "signal":   "Indecision — wait for next candle direction to confirm",
            "strength": "medium",
        }

    # Hammer: lower wick > 2x body, small upper wick
    if lower_wick > 2 * body and upper_wick < body:
        return {
            "detected": True,
            "type":     "hammer",
            "signal":   "Bullish reversal — look for CALLS if at support level",
            "strength": "strong",
        }

    # Inverted hammer: upper wick > 2x body, small lower wick
    if upper_wick > 2 * body and lower_wick < body:
        return {
            "detected": True,
            "type":     "inverted_hammer",
            "signal":   "Bearish reversal — look for PUTS if at resistance level",
            "strength": "strong",
        }

    return {"detected": False, "type": "none", "signal": "No reversal candle"}


# ── VWAP calculation ───────────────────────────────────────────


def get_previous_day_levels(ticker: str) -> dict:
    """Get previous trading day high and low — key resistance/support levels."""
    try:
        df = yf.download(ticker, interval='1d', period='5d', progress=False, auto_adjust=True)
        if df.empty or len(df) < 2:
            return {'high': None, 'low': None}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        prev = df.iloc[-2]
        return {'high': round(float(prev['high']), 2), 'low': round(float(prev['low']), 2), 'date': str(df.index[-2].date())}
    except Exception as e:
        return {'high': None, 'low': None, 'error': str(e)}


def get_key_levels(df: pd.DataFrame, n_levels: int = 5) -> dict:
    """Auto-detect key support/resistance levels and trendlines from swing highs/lows."""
    if df.empty or len(df) < 20:
        return {'resistance': [], 'support': [], 'trendlines': []}

    highs = df['high'].values
    lows  = df['low'].values
    closes = df['close'].values
    window = 5

    swing_highs = []
    for i in range(window, len(highs) - window):
        if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and all(highs[i] >= highs[i+j] for j in range(1, window+1)):
            swing_highs.append(round(float(highs[i]), 2))

    swing_lows = []
    for i in range(window, len(lows) - window):
        if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and all(lows[i] <= lows[i+j] for j in range(1, window+1)):
            swing_lows.append(round(float(lows[i]), 2))

    current_price = float(closes[-1])

    def cluster_levels(levels, pct=0.003):
        if not levels:
            return []
        levels = sorted(set(levels))
        clustered, group = [], [levels[0]]
        for level in levels[1:]:
            if (level - group[0]) / group[0] < pct:
                group.append(level)
            else:
                clustered.append(round(sum(group)/len(group), 2))
                group = [level]
        clustered.append(round(sum(group)/len(group), 2))
        return clustered

    res_levels = cluster_levels(swing_highs)
    sup_levels = cluster_levels(swing_lows)

    resistance = sorted([l for l in res_levels if l > current_price and (l - current_price)/current_price < 0.05])[:n_levels]
    support    = sorted([l for l in sup_levels if l < current_price and (current_price - l)/current_price < 0.05], reverse=True)[:n_levels]

    trendlines = []
    if len(swing_highs) >= 2:
        trendlines.append({'type': 'resistance', 'p1': swing_highs[-2], 'p2': swing_highs[-1], 'color': '#ef4444', 'label': 'Resistance trend'})
    if len(swing_lows) >= 2:
        trendlines.append({'type': 'support', 'p1': swing_lows[-2], 'p2': swing_lows[-1], 'color': '#22c55e', 'label': 'Support trend'})

    return {'resistance': resistance, 'support': support, 'trendlines': trendlines}

def calculate_vwap(df: pd.DataFrame) -> dict:
    """
    Calculate intraday VWAP.
    Price above VWAP = bullish intraday bias → favor calls
    Price below VWAP = bearish intraday bias → favor puts
    """
    if df.empty:
        return {"vwap": None, "position": "unknown"}

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_pv  = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    vwap    = (cum_pv / cum_vol).iloc[-1]
    price   = float(df["close"].iloc[-1])

    pct_from_vwap = (price - vwap) / vwap * 100

    if price > float(vwap):
        position = "above"
        bias     = "bullish"
        signal   = f"Price {pct_from_vwap:.2f}% above VWAP → bullish intraday bias → favor CALLS"
    else:
        position = "below"
        bias     = "bearish"
        signal   = f"Price {abs(pct_from_vwap):.2f}% below VWAP → bearish intraday bias → favor PUTS"

    return {
        "vwap":     round(float(vwap), 2),
        "price":    round(price, 2),
        "position": position,
        "bias":     bias,
        "pct":      round(pct_from_vwap, 2),
        "signal":   signal,
    }


# ── Options signal combiner ────────────────────────────────────

def generate_options_signal(
    structure:    dict,
    vwap:         dict,
    hammer:       dict,
    timeframe:    dict,
    pm_levels:    dict,
    current_price: float,
) -> dict:
    """
    Combine all signals into a final CALLS or PUTS recommendation
    with confidence score.

    Weighting:
      Structure (HH/HL or LH/LL): 40%
      VWAP position:               30%
      Hammer candle at level:      20%
      Time of day phase:           10%
    """
    bull_score = 0.0
    bear_score = 0.0
    reasons    = []

    # Structure signal (40%)
    if structure["bias"] == "bullish":
        bull_score += 0.40
        reasons.append("✅ Uptrend structure (HH + HL)")
    elif structure["bias"] == "bearish":
        bear_score += 0.40
        reasons.append("✅ Downtrend structure (LH + LL)")
    else:
        reasons.append("⚠️ Choppy — no structural bias")

    # VWAP signal (30%)
    if vwap.get("bias") == "bullish":
        bull_score += 0.30
        reasons.append(f"✅ Price above VWAP ({vwap.get('pct', 0):+.2f}%)")
    elif vwap.get("bias") == "bearish":
        bear_score += 0.30
        reasons.append(f"✅ Price below VWAP ({vwap.get('pct', 0):+.2f}%)")

    # Hammer candle (20%)
    if hammer.get("detected"):
        if "hammer" in hammer.get("type", "") and "inverted" not in hammer.get("type", ""):
            bull_score += 0.20
            reasons.append(f"✅ Hammer candle at level → bullish")
        elif "inverted" in hammer.get("type", ""):
            bear_score += 0.20
            reasons.append(f"✅ Inverted hammer at level → bearish")
        elif hammer.get("type") == "doji":
            reasons.append("⚠️ Doji — wait for next candle")

    # Pre-market level proximity (bonus signal)
    if pm_levels.get("high") and pm_levels.get("low"):
        pm_high = pm_levels["high"]
        pm_low  = pm_levels["low"]
        range_  = pm_high - pm_low

        if range_ > 0:
            pct_to_high = abs(current_price - pm_high) / range_
            pct_to_low  = abs(current_price - pm_low)  / range_

            if pct_to_low < 0.10:
                bull_score += 0.10
                reasons.append(f"✅ Price near pre-market LOW (support) → bullish")
            elif pct_to_high < 0.10:
                bear_score += 0.10
                reasons.append(f"✅ Price near pre-market HIGH (resistance) → bearish")

    # Determine final signal
    if bull_score > bear_score and bull_score >= 0.50:
        signal     = "CALLS"
        confidence = bull_score
        color      = "#22c55e"
        action     = f"Buy CALLS — {confidence:.0%} confidence. Wait for pullback entry."
    elif bear_score > bull_score and bear_score >= 0.50:
        signal     = "PUTS"
        confidence = bear_score
        color      = "#ef4444"
        action     = f"Buy PUTS — {confidence:.0%} confidence. Wait for bounce entry."
    else:
        signal     = "WAIT"
        confidence = 0.0
        color      = "#f59e0b"
        action     = "No clear signal. Wait for better setup. Do not force a trade."

    return {
        "signal":     signal,
        "confidence": round(confidence, 2),
        "color":      color,
        "action":     action,
        "reasons":    reasons,
    }


# ── Main endpoint ──────────────────────────────────────────────

@router.get("/{ticker}")
@limiter.limit("30/minute")
async def get_daytrading_signal(
    request: Request,
    ticker: str    = Depends(validate_ticker),
    _key:   str    = Depends(verify_api_key),
):
    """
    Complete day trading signal for options traders.

    Returns:
      - Current recommended timeframe based on market hours
      - Pre-market high/low levels
      - Market structure (uptrend/downtrend/choppy)
      - VWAP position and bias
      - Latest candle pattern detection
      - Final CALLS/PUTS/WAIT signal with confidence
      - Intraday chart data (2m and 5m)
    """
    try:
        # Get timeframe recommendation
        tf = get_recommended_timeframe()

        # Fetch intraday data for multiple timeframes
        # Fetch intraday data for multiple timeframes (parallel for speed)
        import concurrent.futures
        def _fetch(args): return fetch_intraday(*args)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(_fetch, [
                (ticker, "2m",  "1d"),
                (ticker, "5m",  "1d"),
                (ticker, "15m", "1d"),
                (ticker, "1h",  "5d"),
            ]))
        df_2m, df_5m, df_10m, df_1h = results

        # Get pre-market levels
        pm_levels = get_premarket_levels(ticker)

        # Get previous day high/low
        prev_day = get_previous_day_levels(ticker)

        # Get key support/resistance levels and trendlines
        key_levels = get_key_levels(df_5m if not df_5m.empty else df_1h)

        # Current price
        price_df = df_5m if not df_5m.empty else df_1h
        current_price = float(price_df["close"].iloc[-1]) if not price_df.empty else 0

        # Structure detection on primary timeframe
        primary_df = {
            "2m": df_2m, "5m": df_5m, "10m": df_10m, "1H": df_1h
        }.get(tf["primary"], df_5m)

        structure = detect_structure(primary_df if not primary_df.empty else df_5m)

        # VWAP on 5m (whole day)
        vwap = calculate_vwap(df_5m) if not df_5m.empty else {"vwap": None}

        # Hammer detection on primary timeframe
        hammer = detect_hammer_candle(primary_df if not primary_df.empty else df_5m)

        # Generate options signal
        options = generate_options_signal(
            structure, vwap, hammer, tf, pm_levels, current_price
        )

        # Prepare chart data for frontend
        def df_to_chart(df, limit=100):
            if df.empty:
                return []
            return [
                {
                    "time":   int(row.Index.timestamp()),
                    "open":   round(float(row.open),  2),
                    "high":   round(float(row.high),  2),
                    "low":    round(float(row.low),   2),
                    "close":  round(float(row.close), 2),
                    "volume": int(row.volume),
                }
                for row in df.tail(limit).itertuples()
            ]

        return {
            "ticker":        ticker,
            "timestamp":     datetime.now(ET).isoformat(),
            "current_price": current_price,
            "market_phase":  tf,
            "premarket":     pm_levels,
            "structure":     structure,
            "vwap":          vwap,
            "candle":        hammer,
            "options":       options,
            "prev_day":      prev_day,
            "key_levels":    key_levels,
            "charts": {
                "2m":  df_to_chart(df_2m),
                "5m":  df_to_chart(df_5m),
                "10m": df_to_chart(df_10m),
                "1h":  df_to_chart(df_1h, 50),
            },
        }

    except Exception as e:
        raise HTTPException(500, f"Day trading signal error: {str(e)}")