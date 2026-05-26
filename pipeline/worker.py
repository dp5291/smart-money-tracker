"""
pipeline/worker.py — Celery background worker.

Runs scheduled tasks every 5 minutes:
  1. Fetch latest price data for all tickers
  2. Recompute all 25 indicators
  3. Run LSTM inference
  4. Update Redis cache
  5. Check alert conditions and log them

Setup (3 terminal windows):
  Terminal 1: redis-server
  Terminal 2: celery -A pipeline.worker worker --loglevel=info
  Terminal 3: celery -A pipeline.worker beat --loglevel=info
  Terminal 4: uvicorn api.main:app --reload
"""

import os
import sys
import json
import redis
from datetime import datetime
from celery import Celery
from celery.schedules import crontab

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TICKERS, REDIS_URL

# ── Celery app setup ───────────────────────────────────────────
celery_app = Celery(
    "smart_money",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Run update_all_tickers every 5 minutes
    beat_schedule={
        "update-all-tickers": {
            "task": "pipeline.worker.update_all_tickers",
            "schedule": 300.0,  # 300 seconds = 5 minutes
        },
    },
)

# Redis client for caching
r = redis.from_url(REDIS_URL)


@celery_app.task
def update_ticker(ticker: str):
    """
    Update a single ticker:
      1. Fetch new price data
      2. Compute indicators
      3. Run model inference
      4. Cache result in Redis (TTL: 6 minutes)
      5. Check alert conditions
    """
    from data.fetcher import fetch_ohlcv
    from data.indicators import compute_all_indicators, get_feature_matrix
    from data.sentiment import get_combined_sentiment, TICKER_NAMES
    from models.lstm import LSTMPredictor, predict_latest
    from config import FEATURE_COLUMNS, LOOKBACK_DAYS, MODEL_DIR
    import torch, joblib, numpy as np

    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Updating {ticker}...")

    try:
        # Fetch and compute indicators
        df      = fetch_ohlcv(ticker, period="1y")
        df      = compute_all_indicators(df)
        feat_df = get_feature_matrix(df, FEATURE_COLUMNS)
        latest  = df.iloc[-1]

        # Load model
        model_path  = os.path.join(MODEL_DIR, f"{ticker}.pt")
        scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

        if os.path.exists(model_path):
            model = LSTMPredictor(input_size=len(FEATURE_COLUMNS))
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            scaler   = joblib.load(scaler_path)
            features = scaler.transform(feat_df.values)
            pred     = predict_latest(model, features, LOOKBACK_DAYS)
        else:
            pred = {"direction": "neutral", "probability": 0.5, "confidence": 0.5}

        # Get sentiment
        company   = TICKER_NAMES.get(ticker, ticker)
        sentiment = get_combined_sentiment(ticker, company, hours_back=6)

        # Compose cache payload
        payload = {
            "ticker":     ticker,
            "updated_at": datetime.utcnow().isoformat(),
            "price":      float(latest["close"]),
            "prediction": pred,
            "sentiment":  sentiment,
            "indicators": {
                "rsi":          round(float(latest.get("rsi_14", 0.5) or 0.5) * 100, 1),
                "macd":         round(float(latest.get("macd", 0) or 0), 4),
                "golden_cross": bool(latest.get("golden_cross", 0)),
                "bb_width":     round(float(latest.get("bb_width", 0) or 0), 2),
                "volume_ratio": round(float(latest.get("volume_ratio", 1) or 1), 2),
                "sma_200_dist": round(float(latest.get("sma_200_dist", 0) or 0), 2),
            },
        }

        # Store in Redis with 6-minute TTL
        cache_key = f"signal:{ticker}"
        r.setex(cache_key, 360, json.dumps(payload))
        print(f"  Cached {ticker}: {pred['direction']} ({pred['confidence']:.0%} confidence)")

        # Check alert conditions
        check_alerts.delay(ticker, payload)

        return payload

    except Exception as e:
        print(f"  Error updating {ticker}: {e}")
        return {"error": str(e)}


@celery_app.task
def update_all_tickers():
    """Update all tracked tickers in parallel."""
    print(f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] Running scheduled update for {TICKERS}")
    for ticker in TICKERS:
        update_ticker.delay(ticker)


@celery_app.task
def check_alerts(ticker: str, payload: dict):
    """
    Check if alert conditions are met and log them.

    Alert fires when:
      - Model confidence > 70% (strong signal)
      - Sentiment direction AGREES with model direction
      - Not already alerted in the last 4 hours (prevent spam)
    """
    from database import SessionLocal, Alert

    pred      = payload.get("prediction", {})
    sentiment = payload.get("sentiment", {})
    direction = pred.get("direction", "neutral")
    confidence = pred.get("confidence", 0)

    # Only alert on high-confidence non-neutral signals
    if direction == "neutral" or confidence < 0.70:
        return

    # Check if sentiment agrees
    sent_label = sentiment.get("label", "neutral")
    if direction == "bullish" and sent_label not in ("bullish", "neutral"):
        return
    if direction == "bearish" and sent_label not in ("bearish", "neutral"):
        return

    # Check if we already alerted recently (4-hour cooldown)
    cooldown_key = f"alert_cooldown:{ticker}:{direction}"
    if r.exists(cooldown_key):
        return

    # Log the alert
    db = SessionLocal()
    try:
        alert = Alert(
            ticker     = ticker,
            timestamp  = datetime.utcnow(),
            direction  = direction,
            confidence = confidence,
            message    = (
                f"{ticker}: {direction.upper()} signal — "
                f"{confidence:.0%} confidence, "
                f"sentiment: {sent_label}"
            ),
        )
        db.add(alert)
        db.commit()

        # Set cooldown (4 hours = 14400 seconds)
        r.setex(cooldown_key, 14400, "1")
        print(f"  ALERT fired: {alert.message}")

    finally:
        db.close()
