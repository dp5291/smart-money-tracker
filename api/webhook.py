"""
api/webhook.py — Receives TradingView webhook alerts.

Add this to api/main.py by importing it:
    from api.webhook import router as webhook_router
    app.include_router(webhook_router)

TradingView sends a POST request to /webhook/tradingview
with a JSON body every time your Pine Script alert fires.

The endpoint:
  1. Validates the incoming signal
  2. Grabs the latest LSTM prediction for that ticker
  3. Gets the latest FinBERT sentiment
  4. Combines everything into one unified signal
  5. Broadcasts it to all connected WebSocket clients
  6. Logs the alert to the database
"""

import json
import redis
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from config import REDIS_URL, TICKERS

router  = APIRouter(prefix="/webhook", tags=["Webhooks"])
r_cache = redis.from_url(REDIS_URL)


# ── Pydantic model for the incoming TradingView payload ───────

class TradingViewAlert(BaseModel):
    """
    Shape of JSON that TradingView sends to your webhook.
    Matches the alert_message in the Pine Script.
    """
    ticker:        str
    signal:        str          # "bullish" / "bearish" / "squeeze"
    price:         float
    rsi:           Optional[float] = None
    macd:          Optional[float] = None
    sma200_dist:   Optional[float] = None   # % distance from 200 SMA
    volume_ratio:  Optional[float] = None
    golden_cross:  Optional[bool]  = None
    bb_width:      Optional[float] = None
    source:        Optional[str]   = "tradingview"


@router.post("/tradingview")
async def receive_tradingview_alert(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Receive a TradingView webhook alert and process it.

    TradingView sends raw JSON in the body — not always a clean
    Content-Type header — so we parse the body manually.

    Example incoming payload (from Pine Script alert_message):
    {
        "ticker":       "AAPL",
        "signal":       "bullish",
        "price":        189.42,
        "rsi":          58.3,
        "macd":         0.0124,
        "sma200_dist":  12.4,
        "volume_ratio": 2.1,
        "golden_cross": true,
        "bb_width":     4.2,
        "source":       "tradingview"
    }
    """
    # Parse body (TradingView sometimes sends text/plain)
    try:
        body = await request.body()
        data = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(400, "Invalid JSON body from TradingView")

    alert = TradingViewAlert(**data)
    ticker = alert.ticker.upper().replace("NASDAQ:", "").replace("NYSE:", "")

    print(f"\n[TV webhook] {ticker} — {alert.signal.upper()} @ ${alert.price}")
    print(f"  RSI: {alert.rsi}  |  MACD: {alert.macd}")
    print(f"  200 SMA dist: {alert.sma200_dist}%  |  Golden cross: {alert.golden_cross}")

    # Process in background so we return 200 to TradingView immediately
    # (TradingView retries if it doesn't get 200 within 3 seconds)
    background_tasks.add_task(process_alert, ticker, alert)

    return {"status": "received", "ticker": ticker, "signal": alert.signal}


async def process_alert(ticker: str, alert: TradingViewAlert):
    """
    Background task: enrich the TradingView signal with LSTM + sentiment,
    then push to frontend via WebSocket.
    """
    try:
        # Get cached LSTM prediction (computed by Celery every 5 min)
        cached = r_cache.get(f"signal:{ticker}")
        if cached:
            cached_data = json.loads(cached)
            lstm_pred = cached_data.get("prediction", {})
            sentiment = cached_data.get("sentiment", {})
        else:
            lstm_pred = {"direction": "neutral", "probability": 0.5, "confidence": 0.5}
            sentiment = {"combined_score": 0.0, "label": "neutral"}

        # Combine TradingView signal + LSTM + sentiment
        unified = build_unified_signal(ticker, alert, lstm_pred, sentiment)

        # Cache the unified signal
        r_cache.setex(f"unified:{ticker}", 360, json.dumps(unified))

        # Check if this is a high-confidence alert worth broadcasting
        if unified["confidence"] >= 0.65 and unified["direction"] != "neutral":
            await broadcast_signal(ticker, unified)
            log_alert(ticker, unified)

        print(f"  [{ticker}] Unified signal: {unified['direction']} "
              f"({unified['confidence']:.0%} confidence)")

    except Exception as e:
        print(f"  Error processing alert for {ticker}: {e}")


def build_unified_signal(
    ticker:   str,
    tv:       TradingViewAlert,
    lstm:     dict,
    sentiment: dict,
) -> dict:
    """
    Combine TradingView technical signal + LSTM model + sentiment
    into one unified signal with a combined confidence score.

    Weighting:
      - TradingView indicators: 40% (objective technical facts)
      - LSTM model prediction:  40% (pattern recognition on history)
      - FinBERT sentiment:      20% (news/social context)

    This is the core value-add of your project — TradingView gives
    the technical signal, your model gives the ML context, sentiment
    gives the real-world context. No other student project does this.
    """

    # Convert TradingView signal to a numeric score (-1 to +1)
    tv_scores = {"bullish": 1.0, "squeeze": 0.7, "neutral": 0.0, "bearish": -1.0}
    tv_score  = tv_scores.get(tv.signal, 0.0)

    # LSTM score: convert 0-1 probability to -1 to +1
    lstm_prob  = lstm.get("probability", 0.5)
    lstm_score = (lstm_prob - 0.5) * 2   # 0.5 → 0, 1.0 → 1.0, 0.0 → -1.0

    # Sentiment score (-1 to +1, already in this range)
    sent_score = sentiment.get("combined_score", 0.0)

    # Weighted combination
    combined = 0.40 * tv_score + 0.40 * lstm_score + 0.20 * sent_score

    # Determine direction and confidence
    if combined > 0.20:
        direction  = "bullish"
        confidence = min(combined, 1.0)
    elif combined < -0.20:
        direction  = "bearish"
        confidence = min(abs(combined), 1.0)
    else:
        direction  = "neutral"
        confidence = 1.0 - abs(combined) * 5   # inverted: 0.0 = totally neutral

    # Bonus confidence boost when all three agree
    all_agree = (
        tv_score > 0 and lstm_score > 0 and sent_score > 0
        or
        tv_score < 0 and lstm_score < 0 and sent_score < 0
    )
    if all_agree:
        confidence = min(confidence * 1.15, 1.0)

    return {
        "ticker":       ticker,
        "timestamp":    datetime.utcnow().isoformat(),
        "direction":    direction,
        "confidence":   round(confidence, 4),
        "combined_score": round(combined, 4),

        # Component scores (shown on dashboard)
        "tv_score":     round(tv_score, 4),
        "lstm_score":   round(lstm_score, 4),
        "sent_score":   round(sent_score, 4),
        "all_agree":    all_agree,

        # TradingView indicator values (display on dashboard)
        "indicators": {
            "signal":       tv.signal,
            "price":        tv.price,
            "rsi":          tv.rsi,
            "macd":         tv.macd,
            "sma200_dist":  tv.sma200_dist,
            "golden_cross": tv.golden_cross,
            "bb_width":     tv.bb_width,
            "volume_ratio": tv.volume_ratio,
        },

        # LSTM details
        "lstm": {
            "probability": lstm_prob,
            "direction":   lstm.get("direction", "neutral"),
        },

        # Sentiment details
        "sentiment": {
            "score":   sent_score,
            "label":   sentiment.get("label", "neutral"),
        },
    }


async def broadcast_signal(ticker: str, signal: dict):
    """
    Push the unified signal to all WebSocket clients watching this ticker.
    Imports manager from main.py — avoids circular imports.
    """
    try:
        from api.main import manager
        await manager.broadcast(ticker, signal)
    except Exception as e:
        print(f"  WebSocket broadcast error: {e}")


def log_alert(ticker: str, signal: dict):
    """Log a high-confidence alert to the database."""
    try:
        from database import SessionLocal, Alert
        db = SessionLocal()
        alert = Alert(
            ticker     = ticker,
            timestamp  = datetime.utcnow(),
            direction  = signal["direction"],
            confidence = signal["confidence"],
            message    = (
                f"{ticker}: {signal['direction'].upper()} — "
                f"{signal['confidence']:.0%} confidence "
                f"({'all signals agree' if signal['all_agree'] else 'mixed signals'})"
            ),
        )
        db.add(alert)
        db.commit()
        db.close()
    except Exception as e:
        print(f"  DB log error: {e}")


# ── Testing endpoint ───────────────────────────────────────────
@router.post("/test")
async def test_webhook(background_tasks: BackgroundTasks):
    """
    Send a fake TradingView alert to test the pipeline.
    Call this to verify your webhook setup works before
    connecting TradingView.

    Usage: POST http://localhost:8000/webhook/test
    """
    fake_alert = TradingViewAlert(
        ticker       = "AAPL",
        signal       = "bullish",
        price        = 189.42,
        rsi          = 58.3,
        macd         = 0.0124,
        sma200_dist  = 12.4,
        volume_ratio = 2.1,
        golden_cross = True,
        bb_width     = 4.2,
        source       = "test",
    )
    background_tasks.add_task(process_alert, "AAPL", fake_alert)
    return {"status": "test alert queued", "payload": fake_alert.model_dump()}
