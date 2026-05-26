"""
run.py — Master setup and run script.

Run this ONCE to set everything up:
    python run.py --setup

Then to start the full system:
    python run.py --start

Or to train a specific ticker:
    python run.py --train AAPL
"""

import os
import sys
import argparse
import subprocess


def setup():
    """Install everything and initialize the database."""
    print("\n" + "="*55)
    print("  Smart Money Tracker — Setup")
    print("="*55)

    # 1. Install Python packages
    print("\n[1/4] Installing Python packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

    # 2. Create .env from example if it doesn't exist
    print("\n[2/4] Setting up environment variables...")
    if not os.path.exists(".env"):
        import shutil
        shutil.copy(".env.example", ".env")
        print("  Created .env — OPEN THIS FILE and fill in your API keys!")
        print("  Required: DATABASE_URL, NEWSAPI_KEY, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET")
    else:
        print("  .env already exists — skipping")

    # 3. Create model save directory
    print("\n[3/4] Creating directories...")
    os.makedirs("./models/saved", exist_ok=True)
    print("  Created models/saved/")

    # 4. Initialize database
    print("\n[4/4] Initializing database...")
    print("  Make sure PostgreSQL is running and DATABASE_URL in .env is correct")
    try:
        from database import init_db
        init_db()
        print("  Database tables created successfully!")
    except Exception as e:
        print(f"  Database error: {e}")
        print("  Fix your DATABASE_URL in .env and run again")
        return

    print("\n" + "="*55)
    print("  Setup complete!")
    print("="*55)
    print("\nNext steps:")
    print("  1. Fill in .env with your API keys")
    print("  2. Train a model: python run.py --train AAPL")
    print("  3. Start the server: python run.py --start")
    print("\nAPI Keys needed (all free):")
    print("  - newsapi.org    → NEWSAPI_KEY")
    print("  - reddit.com/prefs/apps → REDDIT_CLIENT_ID + SECRET")
    print("  - alphavantage.co → ALPHA_VANTAGE_KEY (optional)")


def train_ticker(ticker: str):
    """Train the LSTM model for a specific ticker."""
    print(f"\nTraining {ticker}...")
    from models.train import train_model
    metrics = train_model(ticker)
    print(f"\nTraining complete! Metrics: {metrics}")
    print(f"\nNext: python run.py --backtest {ticker}")


def backtest_ticker(ticker: str):
    """Run backtest for a specific ticker."""
    print(f"\nRunning backtest for {ticker}...")
    from models.backtest import run_backtest, print_backtest_report
    results = run_backtest(ticker)
    print_backtest_report(results)


def start():
    """Print instructions for starting all services."""
    print("\n" + "="*55)
    print("  Starting Smart Money Tracker")
    print("="*55)
    print("\nOpen 4 terminal windows and run one command in each:\n")
    print("  Terminal 1 — Redis:")
    print("    redis-server")
    print()
    print("  Terminal 2 — Celery worker (background tasks):")
    print("    celery -A pipeline.worker worker --loglevel=info")
    print()
    print("  Terminal 3 — Celery beat (scheduler — runs every 5 min):")
    print("    celery -A pipeline.worker beat --loglevel=info")
    print()
    print("  Terminal 4 — FastAPI server:")
    print("    uvicorn api.main:app --reload --port 8000")
    print()
    print("  Then open: http://localhost:8000/docs")
    print("  To test a signal: http://localhost:8000/signal/AAPL")
    print()
    print("  Or start the API only (no background tasks):")
    os.execv(
        sys.executable,
        [sys.executable, "-m", "uvicorn", "api.main:app", "--reload", "--port", "8000"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Money Tracker")
    parser.add_argument("--setup",    action="store_true",  help="Install and initialize")
    parser.add_argument("--train",    type=str, metavar="TICKER", help="Train model for ticker")
    parser.add_argument("--backtest", type=str, metavar="TICKER", help="Backtest ticker")
    parser.add_argument("--start",    action="store_true",  help="Start the API server")
    parser.add_argument("--test",     type=str, metavar="TICKER", help="Quick indicator test")
    args = parser.parse_args()

    if args.setup:
        setup()
    elif args.train:
        train_ticker(args.train.upper())
    elif args.backtest:
        backtest_ticker(args.backtest.upper())
    elif args.start:
        start()
    elif args.test:
        # Quick test — just compute indicators for one ticker
        ticker = args.test.upper()
        print(f"\nTesting indicator computation for {ticker}...")
        from data.fetcher import fetch_ohlcv
        from data.indicators import compute_all_indicators
        df = fetch_ohlcv(ticker, period="1y")
        df = compute_all_indicators(df)
        print(f"\nLatest indicators for {ticker}:")
        latest = df.iloc[-1]
        print(f"  Close:       ${latest['close']:.2f}")
        print(f"  200 SMA:     ${latest.get('sma_200', 0):.2f}  ({latest.get('sma_200_dist', 0):.1f}% away)")
        print(f"  Golden Cross:{bool(latest.get('golden_cross', 0))}")
        print(f"  RSI:         {latest.get('rsi_14', 0.5)*100:.1f}")
        print(f"  BB Width:    {latest.get('bb_width', 0):.2f}%")
        print(f"  Volume ratio:{latest.get('volume_ratio', 1):.2f}x")
    else:
        parser.print_help()
