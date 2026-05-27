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
config.py — Central configuration.
All settings come from environment variables (loaded from .env).
Never hardcode secrets in code.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ──────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/smartmoney")

# ── Redis ─────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── API Keys ──────────────────────────────────────────────────
NEWSAPI_KEY         = os.getenv("NEWSAPI_KEY", "")
REDDIT_CLIENT_ID    = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET= os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT   = os.getenv("REDDIT_USER_AGENT", "SmartMoneyTracker/1.0")
ALPHA_VANTAGE_KEY   = os.getenv("ALPHA_VANTAGE_KEY", "")

# ── Tickers ───────────────────────────────────────────────────
TICKERS = os.getenv("TICKERS", "AAPL,NVDA,TSLA,BTC-USD").split(",")

# ── Model ─────────────────────────────────────────────────────
MODEL_DIR              = os.getenv("MODEL_DIR", "./models/saved")
LOOKBACK_DAYS          = int(os.getenv("LOOKBACK_DAYS", "60"))
PREDICTION_THRESHOLD_BULL = float(os.getenv("PREDICTION_THRESHOLD_BULL", "0.60"))
PREDICTION_THRESHOLD_BEAR = float(os.getenv("PREDICTION_THRESHOLD_BEAR", "0.40"))

# ── LSTM Hyperparameters ───────────────────────────────────────
LSTM_INPUT_SIZE  = 25    # number of features
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS  = 2
LSTM_DROPOUT     = 0.2
LSTM_LR          = 0.001
LSTM_EPOCHS      = 50
LSTM_BATCH_SIZE  = 32

# ── Feature names (in exact order fed to model) ───────────────
FEATURE_COLUMNS = [
    "close_norm",        # normalized closing price
    "volume_norm",       # normalized volume
    "sma_200_dist",      # % distance from 200 SMA  ← KEY FEATURE
    "sma_50_dist",       # % distance from 50 SMA
    "ema_20_dist",       # % distance from 20 EMA
    "golden_cross",      # 1 if 50 SMA > 200 SMA, else 0
    "rsi_14",            # RSI (0-100, normalized to 0-1)
    "macd",              # MACD line
    "macd_signal",       # MACD signal line
    "macd_hist",         # MACD histogram
    "stoch_k",           # Stochastic %K
    "stoch_d",           # Stochastic %D
    "bb_upper_dist",     # % distance to upper Bollinger Band
    "bb_lower_dist",     # % distance to lower Bollinger Band
    "bb_width",          # Bollinger Band width %
    "atr_norm",          # ATR normalized by price
    "vwap_dist",         # % distance from VWAP
    "obv_norm",          # OBV normalized
    "volume_ratio",      # today's volume / 20-day avg volume
    "volume_trend",      # slope of volume over 5 days
    "support_dist",      # % distance from nearest support level
    "resistance_dist",   # % distance from nearest resistance level
    "fib_618_dist",      # % distance from 61.8% Fibonacci level
    "roc_10",            # 10-day Rate of Change
    "finbert_score",     # FinBERT sentiment (-1 to +1)
]
