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
database.py — Database setup with SQLAlchemy.

Tables:
  price_data     — OHLCV + all 25 computed indicators per ticker per day
  predictions    — model output per ticker per run
  sentiment_data — FinBERT scores per ticker per collection window
  alerts         — fired alerts log

TimescaleDB (optional but recommended):
  If installed, run: SELECT create_hypertable('price_data', 'timestamp');
  This makes time-range queries 10-100x faster on large datasets.
"""

from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    DateTime, Boolean, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from config import DATABASE_URL

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ── Models ─────────────────────────────────────────────────────

class PriceData(Base):
    """One row = one trading day for one ticker, with all 25 indicators."""
    __tablename__ = "price_data"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    ticker        = Column(String(20), nullable=False, index=True)
    timestamp     = Column(DateTime, nullable=False, index=True)
    open          = Column(Float)
    high          = Column(Float)
    low           = Column(Float)
    close         = Column(Float)
    volume        = Column(Float)

    # ── Trend indicators ──────────────────────────────────────
    sma_200       = Column(Float)
    sma_50        = Column(Float)
    ema_20        = Column(Float)
    sma_200_dist  = Column(Float)   # (close - sma_200) / sma_200 * 100
    sma_50_dist   = Column(Float)
    ema_20_dist   = Column(Float)
    golden_cross  = Column(Boolean) # True if sma_50 > sma_200

    # ── Momentum indicators ───────────────────────────────────
    rsi_14        = Column(Float)
    macd          = Column(Float)
    macd_signal   = Column(Float)
    macd_hist     = Column(Float)
    stoch_k       = Column(Float)
    stoch_d       = Column(Float)
    roc_10        = Column(Float)

    # ── Volatility indicators ─────────────────────────────────
    bb_upper      = Column(Float)
    bb_lower      = Column(Float)
    bb_middle     = Column(Float)
    bb_width      = Column(Float)
    bb_upper_dist = Column(Float)
    bb_lower_dist = Column(Float)
    atr_14        = Column(Float)
    atr_norm      = Column(Float)   # atr / close

    # ── Volume indicators ─────────────────────────────────────
    vwap          = Column(Float)
    vwap_dist     = Column(Float)
    obv           = Column(Float)
    obv_norm      = Column(Float)
    volume_ratio  = Column(Float)   # volume / 20-day avg volume
    volume_trend  = Column(Float)   # 5-day volume slope

    # ── Structure / pattern ───────────────────────────────────
    support       = Column(Float)
    resistance    = Column(Float)
    support_dist  = Column(Float)
    resistance_dist = Column(Float)
    fib_618       = Column(Float)
    fib_618_dist  = Column(Float)

    created_at    = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """Model output per ticker per inference run."""
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    ticker          = Column(String(20), nullable=False, index=True)
    timestamp       = Column(DateTime, nullable=False)
    prediction      = Column(Float)         # 0.0 – 1.0 (sigmoid output)
    direction       = Column(String(10))    # "bullish" / "bearish" / "neutral"
    confidence      = Column(Float)         # same as prediction but renamed
    finbert_score   = Column(Float)
    created_at      = Column(DateTime, default=datetime.utcnow)


class SentimentData(Base):
    """FinBERT sentiment scores collected from news + Reddit."""
    __tablename__ = "sentiment_data"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    ticker        = Column(String(20), nullable=False, index=True)
    timestamp     = Column(DateTime, nullable=False)
    score         = Column(Float)    # -1.0 (very bearish) to +1.0 (very bullish)
    article_count = Column(Integer)
    source        = Column(String(50))   # "news" / "reddit" / "combined"
    created_at    = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    """Log of every alert that fired."""
    __tablename__ = "alerts"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    ticker      = Column(String(20), nullable=False)
    timestamp   = Column(DateTime, nullable=False)
    direction   = Column(String(10))
    confidence  = Column(Float)
    message     = Column(Text)
    sent        = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables. Run once on first launch."""
    Base.metadata.create_all(engine)
    print("Database tables created.")


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
