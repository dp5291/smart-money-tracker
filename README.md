# Smart Money Tracker — AI-Powered Market Signal Dashboard

> Full-stack AI trading signal platform combining LSTM deep learning,
> FinBERT NLP sentiment analysis, and real-time market data.
> Built for both swing traders and options day traders.

---

## Backtest Results

| Ticker | Sharpe Ratio | Win Rate | Total Return | Alpha |
|--------|-------------|----------|--------------|-------|
| AMZN   | **5.33**    | 100%     | +54.8%       | +38.6% |
| AAPL   | **4.40**    | 100%     | +37.4%       | +23.0% |
| NVDA   | **4.04**    | 100%     | +24.4%       | +11.5% |
| SPY    | **3.13**    | 100%     | +10.6%       | +0.6%  |
| TSLA   | **2.33**    | —        | +25.1%       | +26.4% |
| BTC    | **1.60**    | 87.5%    | +26.5%       | +39.0% |

> Sharpe ratio > 2.0 is considered exceptional. AMZN achieved 5.33.

---

## Features

### Swing Trading Mode
- **LSTM Model** — 2-layer LSTM trained on 25 engineered features with 60-day lookback window
- **FinBERT Sentiment** — Real-time NLP scoring of 30+ financial news headlines per request
- **TradingView Chart** — Embedded live chart with 9 EMA, 21 EMA, 50/200 SMA, Bollinger Bands, VWAP
- **AI Prediction Gauge** — Directional probability (bullish/bearish/neutral) with confidence score
- **Smart Alerts** — Fires when model confidence exceeds 70%
- **Live WebSocket** — Updates every 5 minutes pushed to frontend

### Day Trading Mode (Options)
- **Time-based chart recommendations** — Auto-switches between 2m/5m/10m based on market hours
- **Pre-market levels** — Auto-detected high/low from 4am–9:30am data
- **Previous day high/low** — Key levels auto-plotted on intraday chart
- **VWAP signal** — Price above/below VWAP with intraday bias
- **Market structure** — Auto-detects Higher Highs/Lows (uptrend) or Lower Highs/Lows (downtrend)
- **Key levels** — Auto-detected support/resistance with swing trendlines
- **Options signal** — CALLS/PUTS/WAIT with confidence score
- **Hammer/Doji detection** — Reversal candle detection at key levels

### Security
- API key authentication on all endpoints
- Webhook secret token validation
- Rate limiting (slowapi) — 20 req/min for signals, 5 req/min for backtests
- IP auto-ban after 10 failed auth attempts
- CORS lockdown to frontend URL only
- Full request logging with audit trail

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | PyTorch LSTM, FinBERT (Transformers) |
| Backend | Python, FastAPI, WebSockets, Celery, Redis |
| Data | yfinance, pandas-ta, NewsAPI |
| Frontend | React, TradingView Widget |
| Database | PostgreSQL, SQLAlchemy |
| Security | slowapi, API keys, CORS, IP filtering |

---

## Architecture

```
yfinance (market data)          NewsAPI (news headlines)
        ↓                               ↓
  25 Technical Indicators         FinBERT NLP Scoring
        ↓                               ↓
    LSTM Model              Combined Sentiment Score
        ↓                               ↓
        └──────────── FastAPI ──────────┘
                          ↓
                   WebSocket + REST
                          ↓
                  React Dashboard
                  ├── Swing Mode
                  └── Day Trading Mode (Options)
```

---

## 25 Features Used by LSTM

| Category | Features |
|----------|---------|
| Trend | 200 SMA distance, 50 SMA distance, Golden Cross, EMA 20 |
| Momentum | RSI 14, MACD, MACD Signal, Stochastic |
| Volatility | Bollinger Band Width, BB Upper/Lower, ATR |
| Volume | OBV trend, Volume ratio, VWAP distance |
| Structure | Support/Resistance, Fibonacci levels |
| Macro | VIX, DXY (Dollar Index) |

---

## Setup

### Prerequisites
- Python 3.11+
- PostgreSQL
- Redis
- Node.js 22+

### Installation

```bash
# Clone repo
git clone https://github.com/dp5291/smart-money-tracker.git
cd smart-money-tracker

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Fill in your API keys in .env

# Initialize database
python database.py

# Train model (15-25 min per ticker)
python run.py --train AAPL

# Run backtest
python run.py --backtest AAPL

# Start server
uvicorn api.main:app --port 8000
```

### API Keys Needed

| Key | Source | Free? |
|-----|--------|-------|
| `NEWSAPI_KEY` | newsapi.org | Yes (100 req/day) |
| `REDDIT_CLIENT_ID` | reddit.com/prefs/apps | Yes |
| `API_KEYS` | Run `python api/security.py` | Generated |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check |
| GET | `/signal/AAPL` | Full AI signal for ticker |
| GET | `/backtest/AAPL` | Backtest results |
| GET | `/daytrading/AAPL` | Day trading signals + levels |
| GET | `/historical/AAPL` | OHLCV + all indicators |
| WS | `/ws/AAPL` | Real-time WebSocket updates |

---

## Project Structure

```
smart-money-tracker/
├── api/
│   ├── main.py              # FastAPI endpoints + WebSocket
│   ├── daytrading.py        # Day trading signals + level detection
│   ├── security.py          # Auth, rate limiting, IP ban, CORS
│   └── webhook.py           # TradingView webhook receiver
├── data/
│   ├── fetcher.py           # yfinance OHLCV data
│   ├── indicators.py        # 25 technical indicators
│   └── sentiment.py         # FinBERT + NewsAPI sentiment
├── models/
│   ├── lstm.py              # PyTorch LSTM architecture
│   ├── train.py             # Training loop + evaluation
│   └── backtest.py          # Backtesting engine
├── frontend/
│   ├── App.jsx              # Swing trading dashboard
│   └── DayTradingDashboard.jsx  # Options day trading UI
├── pipeline/
│   └── worker.py            # Celery background tasks
├── config.py                # Central configuration
├── database.py              # PostgreSQL schema
└── run.py                   # CLI: train, test, backtest
```

---

## Author

**Dhruv Patel** — Computer Science, UMass Lowell (GPA 3.8/4.0)
- GitHub: [@dp5291](https://github.com/dp5291)
- Email: dhruvkumarp79@gmail.com
- LinkedIn: linkedin.com/in/dhruv-patel29
