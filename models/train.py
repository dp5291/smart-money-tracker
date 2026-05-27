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
models/train.py — Train the LSTM model on historical data.

How to run:
    python models/train.py --ticker AAPL
    python models/train.py --ticker NVDA --epochs 100

What it does:
  1. Fetches 2 years of AAPL data
  2. Computes all 25 indicators
  3. Gets FinBERT sentiment (or uses 0 if not configured)
  4. Splits: 70% train, 15% validation, 15% test
  5. Trains 2-layer LSTM with early stopping
  6. Saves best model to models/saved/<ticker>.pt
  7. Prints final metrics: accuracy, precision, recall on test set
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    FEATURE_COLUMNS, LOOKBACK_DAYS,
    LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LSTM_LR, LSTM_EPOCHS, LSTM_BATCH_SIZE, MODEL_DIR
)
from data.fetcher import fetch_ohlcv
from data.indicators import compute_all_indicators, get_feature_matrix
from models.lstm import LSTMPredictor, StockDataset


def train_model(ticker: str, epochs: int = LSTM_EPOCHS) -> dict:
    """
    Full training pipeline for one ticker.

    Returns:
        dict with final test metrics: accuracy, precision, recall, f1
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {ticker} on {device} | Epochs: {epochs}")

    # ── Step 1: Data ──────────────────────────────────────────
    print("\n[1/5] Fetching and processing data...")
    df = fetch_ohlcv(ticker, period="5y")       # 5 years for more training data
    df = compute_all_indicators(df)

    close_prices = df["close"].values
    feature_df   = get_feature_matrix(df, FEATURE_COLUMNS)
    features     = feature_df.values

    print(f"  Dataset: {len(features)} rows × {features.shape[1]} features")
    print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")

    # ── Step 2: Scale features ────────────────────────────────
    # MinMaxScaler scales each feature to [0, 1] range
    # Important: fit ONLY on training data to prevent data leakage
    n = len(features)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    scaler = MinMaxScaler(feature_range=(0, 1))
    features[:train_end] = scaler.fit_transform(features[:train_end])
    features[train_end:] = scaler.transform(features[train_end:])

    # Save scaler — needed for inference
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")

    # ── Step 3: Create datasets ───────────────────────────────
    print("\n[2/5] Creating datasets...")
    train_feat = features[:train_end]
    val_feat   = features[train_end:val_end]
    test_feat  = features[val_end:]

    train_close = close_prices[:train_end]
    val_close   = close_prices[train_end:val_end]
    test_close  = close_prices[val_end:]

    train_dataset = StockDataset(train_feat, train_close, LOOKBACK_DAYS)
    val_dataset   = StockDataset(val_feat,   val_close,   LOOKBACK_DAYS)
    test_dataset  = StockDataset(test_feat,  test_close,  LOOKBACK_DAYS)

    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=LSTM_BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=LSTM_BATCH_SIZE, shuffle=False)

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ── Step 4: Model setup ───────────────────────────────────
    print("\n[3/5] Setting up model...")
    model = LSTMPredictor(
        input_size  = len(FEATURE_COLUMNS),
        hidden_size = LSTM_HIDDEN_SIZE,
        num_layers  = LSTM_NUM_LAYERS,
        dropout     = LSTM_DROPOUT,
    ).to(device)

    # Class weighting: handle imbalance (markets go up ~53% of days)
    n_pos = sum(train_dataset.y)
    n_neg = len(train_dataset.y) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # ── Step 5: Training loop ─────────────────────────────────
    print("\n[4/5] Training...")
    best_val_loss  = float("inf")
    patience_count = 0
    patience_limit = 10    # stop if val loss doesn't improve for 10 epochs
    model_path     = os.path.join(MODEL_DIR, f"{ticker}.pt")

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            # Use raw logits with BCEWithLogitsLoss (more numerically stable)
            out = model.lstm(X_batch)[0][:, -1, :]
            out = model.fc2(model.relu(model.fc1(model.dropout(out))))
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent explosion
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model.lstm(X_batch)[0][:, -1, :]
                out = model.fc2(model.relu(model.fc1(model.dropout(out))))
                val_loss += criterion(out, y_batch).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        # Early stopping + save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1
            if patience_count >= patience_limit:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  Best model saved to {model_path}")

    # ── Step 6: Test set evaluation ───────────────────────────
    print("\n[5/5] Evaluating on test set...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            probs   = model(X_batch).squeeze().cpu().numpy()
            labels  = y_batch.squeeze().numpy()
            all_preds.extend(probs if probs.ndim > 0 else [probs])
            all_labels.extend(labels if labels.ndim > 0 else [labels])

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Convert probabilities to binary predictions at 0.5 threshold
    binary_preds = (all_preds > 0.5).astype(int)

    tp = np.sum((binary_preds == 1) & (all_labels == 1))
    tn = np.sum((binary_preds == 0) & (all_labels == 0))
    fp = np.sum((binary_preds == 1) & (all_labels == 0))
    fn = np.sum((binary_preds == 0) & (all_labels == 1))

    accuracy  = (tp + tn) / len(all_labels)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    metrics = {
        "ticker":    ticker,
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "test_size": len(all_labels),
    }

    print("\n" + "="*40)
    print(f"  {ticker} Test Results")
    print("="*40)
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Test size: {len(all_labels)} days")
    print("="*40)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Smart Money Tracker LSTM")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker to train on")
    parser.add_argument("--epochs", type=int, default=LSTM_EPOCHS, help="Max training epochs")
    args = parser.parse_args()

    metrics = train_model(args.ticker, args.epochs)
    print(f"\nDone. Model saved to {MODEL_DIR}/{args.ticker}.pt")
