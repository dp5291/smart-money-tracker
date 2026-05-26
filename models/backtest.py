"""
models/backtest.py — Backtesting engine.

Simulates trading with model signals on historical data.
Computes professional-grade metrics used by hedge funds:
  - Win rate
  - Sharpe ratio (reward per unit of risk)
  - Maximum drawdown (worst peak-to-trough loss)
  - Total return
  - Monthly returns heatmap data

How to run:
    python models/backtest.py --ticker AAPL

What a good result looks like:
    Win rate:    > 55%   (random is 50%)
    Sharpe:      > 1.0   (> 2.0 is excellent)
    Max drawdown: < 20%  (lower is better)
    Total return: beats buy-and-hold
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import FEATURE_COLUMNS, LOOKBACK_DAYS, MODEL_DIR
from data.fetcher import fetch_ohlcv
from data.indicators import compute_all_indicators, get_feature_matrix
from models.lstm import LSTMPredictor, predict_latest


def run_backtest(
    ticker:          str,
    bull_threshold:  float = 0.60,
    bear_threshold:  float = 0.40,
    commission:      float = 0.001,   # 0.1% per trade (realistic)
    initial_capital: float = 10_000,
) -> dict:
    """
    Simulate trading the model's signals on historical test data.

    Strategy:
      - BUY (go long) when prediction > bull_threshold
      - SELL (go to cash) when prediction < bear_threshold
      - Hold when between thresholds (neutral zone)

    Args:
        ticker:          Stock to backtest
        bull_threshold:  Minimum confidence to buy (0.60 = 60%)
        bear_threshold:  Maximum confidence to sell (0.40 = 40%)
        commission:      Transaction cost per trade (0.1%)
        initial_capital: Starting portfolio value

    Returns:
        Full backtest results dict with all metrics and trade log.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model and scaler ─────────────────────────────────
    model_path  = os.path.join(MODEL_DIR, f"{ticker}.pt")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            f"Run: python models/train.py --ticker {ticker}"
        )

    model = LSTMPredictor(input_size=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    scaler = joblib.load(scaler_path)

    # ── Prepare data (same split as training) ─────────────────
    df       = fetch_ohlcv(ticker, period="5y")
    df       = compute_all_indicators(df)
    feat_df  = get_feature_matrix(df, FEATURE_COLUMNS)
    features = scaler.transform(feat_df.values)
    closes   = df["close"].values
    dates    = df.index

    # Use the TEST portion only (last 15%)
    n        = len(features)
    test_start = int(n * 0.85)
    test_features = features[test_start:]
    test_closes   = closes[test_start:]
    test_dates    = dates[test_start:]

    # ── Run predictions day by day ────────────────────────────
    predictions = []
    # We need enough lookback history before the test period starts
    full_features = features   # use full history for lookback window

    for i in range(LOOKBACK_DAYS, len(test_features)):
        # The window for this day pulls from the full history
        global_idx = test_start + i
        if global_idx < LOOKBACK_DAYS:
            continue
        window = full_features[global_idx - LOOKBACK_DAYS : global_idx]
        x = torch.tensor(window[np.newaxis].astype(np.float32)).to(device)
        with torch.no_grad():
            prob = model(x).item()
        predictions.append(prob)

    # Align with the correct dates and prices
    pred_start   = LOOKBACK_DAYS
    pred_closes  = test_closes[pred_start:]
    pred_dates   = test_dates[pred_start:]
    predictions  = np.array(predictions[:len(pred_closes)])

    # ── Simulate portfolio ────────────────────────────────────
    capital       = initial_capital
    in_position   = False
    entry_price   = 0.0
    trades        = []
    daily_values  = [capital]

    for i in range(1, len(predictions)):
        prob        = predictions[i]
        today_close = pred_closes[i]
        today_date  = pred_dates[i]

        if not in_position and prob > bull_threshold:
            # BUY signal
            shares     = capital / today_close
            commission_cost = capital * commission
            capital   -= commission_cost
            in_position = True
            entry_price = today_close
            trades.append({
                "date":   today_date,
                "action": "BUY",
                "price":  today_close,
                "prob":   prob,
            })

        elif in_position and prob < bear_threshold:
            # SELL signal
            exit_value  = shares * today_close
            commission_cost = exit_value * commission
            pnl_pct     = (today_close - entry_price) / entry_price * 100
            capital     = exit_value - commission_cost
            in_position = False
            trades.append({
                "date":   today_date,
                "action": "SELL",
                "price":  today_close,
                "prob":   prob,
                "pnl_pct": round(pnl_pct, 2),
            })

        # Track daily portfolio value
        if in_position:
            portfolio_val = shares * today_close
        else:
            portfolio_val = capital
        daily_values.append(portfolio_val)

    # Close any open position at the end
    if in_position and len(pred_closes) > 0:
        final_price = pred_closes[-1]
        capital = shares * final_price * (1 - commission)

    # ── Compute metrics ───────────────────────────────────────
    daily_values  = np.array(daily_values[:len(pred_closes)])
    daily_returns = np.diff(daily_values) / daily_values[:-1]

    # Total return
    total_return = (daily_values[-1] - initial_capital) / initial_capital * 100

    # Buy-and-hold return (benchmark)
    bh_return = (pred_closes[-1] - pred_closes[0]) / pred_closes[0] * 100

    # Sharpe ratio (annualized)
    # Risk-free rate assumed 5% annually ≈ 0.013% daily
    rf_daily = 0.05 / 252
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() - rf_daily) / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    # Maximum drawdown
    peak    = np.maximum.accumulate(daily_values)
    drawdown = (daily_values - peak) / peak * 100
    max_dd  = drawdown.min()

    # Win rate
    sell_trades = [t for t in trades if t["action"] == "SELL" and "pnl_pct" in t]
    if sell_trades:
        wins     = sum(1 for t in sell_trades if t["pnl_pct"] > 0)
        win_rate = wins / len(sell_trades) * 100
    else:
        win_rate = 0.0

    # Monthly returns for heatmap
    monthly = {}
    for i, date in enumerate(pred_dates[:len(daily_values)]):
        key = (date.year, date.month)
        monthly[key] = daily_values[i]

    monthly_returns = {}
    prev_val  = initial_capital
    for key in sorted(monthly.keys()):
        curr_val = monthly[key]
        monthly_returns[key] = round((curr_val - prev_val) / prev_val * 100, 2)
        prev_val = curr_val

    results = {
        "ticker":          ticker,
        "total_return":    round(total_return, 2),
        "bh_return":       round(bh_return, 2),
        "sharpe":          round(sharpe, 3),
        "max_drawdown":    round(max_dd, 2),
        "win_rate":        round(win_rate, 2),
        "n_trades":        len(sell_trades),
        "final_capital":   round(daily_values[-1], 2),
        "initial_capital": initial_capital,
        "test_days":       len(pred_closes),
        "daily_values":    daily_values.tolist(),
        "daily_dates":     [str(d.date()) for d in pred_dates[:len(daily_values)]],
        "monthly_returns": {f"{k[0]}-{k[1]:02d}": v for k, v in monthly_returns.items()},
        "trades":          trades[-20:],  # last 20 trades
    }
    return results


def print_backtest_report(results: dict):
    """Print a formatted backtest report to the console."""
    print("\n" + "="*50)
    print(f"  BACKTEST RESULTS — {results['ticker']}")
    print("="*50)
    print(f"  Period:         {results['test_days']} trading days")
    print(f"  Initial capital: ${results['initial_capital']:,.0f}")
    print(f"  Final capital:   ${results['final_capital']:,.0f}")
    print()
    print(f"  Total return:    {results['total_return']:+.1f}%")
    print(f"  Buy & hold:      {results['bh_return']:+.1f}%")
    alpha = results['total_return'] - results['bh_return']
    print(f"  Alpha (outperf): {alpha:+.1f}%")
    print()
    print(f"  Sharpe ratio:    {results['sharpe']:.3f}  (> 1.0 is good, > 2.0 is great)")
    print(f"  Win rate:        {results['win_rate']:.1f}%")
    print(f"  Max drawdown:    {results['max_drawdown']:.1f}%")
    print(f"  Number of trades:{results['n_trades']}")
    print("="*50)

    if results['n_trades'] > 0:
        print(f"\n  Last {min(5, len(results['trades']))} trades:")
        for t in results["trades"][-5:]:
            action = t["action"]
            pnl    = f"  PnL: {t['pnl_pct']:+.1f}%" if "pnl_pct" in t else ""
            print(f"    {t['date'].date() if hasattr(t['date'],'date') else t['date']} "
                  f"{action} @ ${t['price']:.2f} "
                  f"(model: {t['prob']:.1%}){pnl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--capital", type=float, default=10000)
    args = parser.parse_args()

    results = run_backtest(args.ticker, initial_capital=args.capital)
    print_backtest_report(results)
