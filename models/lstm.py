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
models/lstm.py — 2-layer LSTM directional predictor in PyTorch.

Architecture:
  Input:  (batch, sequence=60, features=25)
  LSTM 1: 128 hidden units
  LSTM 2: 64 hidden units
  Dropout: 0.2
  Dense:  64 → 1
  Output: sigmoid probability (> 0.6 = bullish, < 0.4 = bearish)

The model predicts: will the stock close HIGHER tomorrow than today?
It outputs a probability — we threshold it to get buy/sell/neutral signals.

Why 2-layer LSTM:
  Layer 1 captures short-term patterns (RSI divergence, volume spikes)
  Layer 2 captures longer-term context (trend regime, Fib level approach)
  Two layers give the model capacity to learn these hierarchical patterns.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class LSTMPredictor(nn.Module):
    """
    2-layer LSTM for stock direction prediction.

    Args:
        input_size:  Number of features per time step (25)
        hidden_size: Hidden units in LSTM layers (128)
        num_layers:  Number of stacked LSTM layers (2)
        dropout:     Dropout between LSTM layers (0.2)
    """

    def __init__(
        self,
        input_size:  int = 25,
        hidden_size: int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # 2-layer LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,       # input shape: (batch, seq, features)
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Additional dropout before final layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers: compress hidden → output
        self.fc1    = nn.Linear(hidden_size, 64)
        self.relu   = nn.ReLU()
        self.fc2    = nn.Linear(64, 1)

        # Sigmoid: output is a probability between 0 and 1
        self.sigmoid = nn.Sigmoid()

        # Initialize weights with Xavier uniform (helps training stability)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, sequence_length, input_size)
               e.g. (32, 60, 25) — 32 samples, 60 days lookback, 25 features

        Returns:
            Tensor of shape (batch_size, 1) — probability of price going up
        """
        # LSTM pass — we only use the final hidden state
        lstm_out, _ = self.lstm(x)          # (batch, seq, hidden_size)
        last_hidden  = lstm_out[:, -1, :]   # (batch, hidden_size) — last timestep

        # Fully connected layers
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.sigmoid(out)             # (batch, 1)

        return out


# ── Dataset class ──────────────────────────────────────────────

class StockDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for the LSTM.

    Creates sliding windows:
      Given N days of data, creates (N - lookback) samples.
      Each sample X: (lookback, n_features) — e.g. 60 days of 25 features
      Each label  y: 1 if next day close > today close, else 0

    Args:
        features:    np.ndarray of shape (N, n_features)
        close_prices: np.ndarray of shape (N,) — raw close prices for labeling
        lookback:    number of past days to use as input (60)
    """

    def __init__(
        self,
        features:     np.ndarray,
        close_prices: np.ndarray,
        lookback:     int = 60,
    ):
        self.X = []
        self.y = []

        for i in range(lookback, len(features) - 1):
            # Input: last `lookback` rows of features
            self.X.append(features[i - lookback : i])

            # Label: did price go up the NEXT day?
            # We use a 0.5% threshold to avoid labeling noise as signal
            next_close = close_prices[i + 1]
            curr_close = close_prices[i]
            label = 1.0 if (next_close - curr_close) / curr_close > 0.005 else 0.0
            self.y.append(label)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx]),           # (lookback, n_features)
            torch.tensor(self.y[idx]).unsqueeze(0)  # (1,)
        )


# ── Inference helper ───────────────────────────────────────────

def predict_latest(
    model:        LSTMPredictor,
    features:     np.ndarray,
    lookback:     int = 60,
    device:       str = "cpu",
) -> dict:
    """
    Run inference on the most recent data to get today's signal.

    Args:
        model:    trained LSTMPredictor
        features: np.ndarray of shape (N, 25) — full feature history
        lookback: must match what the model was trained with

    Returns:
        {
            "probability": float (0-1),
            "direction":   "bullish" / "bearish" / "neutral",
            "confidence":  float (0-1),
        }

    Usage:
        from models.lstm import LSTMPredictor, predict_latest
        model = LSTMPredictor()
        model.load_state_dict(torch.load("models/saved/aapl.pt"))
        signal = predict_latest(model, features_array)
        print(signal)
    """
    model.eval()
    model.to(device)

    # Take the last `lookback` rows as input
    window = features[-lookback:].astype(np.float32)     # (60, 25)
    x      = torch.tensor(window).unsqueeze(0).to(device)  # (1, 60, 25)

    with torch.no_grad():
        prob = model(x).item()   # scalar between 0 and 1

    # Determine direction based on thresholds
    if prob > 0.60:
        direction = "bullish"
        confidence = prob
    elif prob < 0.40:
        direction = "bearish"
        confidence = 1.0 - prob
    else:
        direction = "neutral"
        confidence = 1.0 - abs(prob - 0.5) * 2   # how neutral (0 = perfectly neutral)

    return {
        "probability": round(prob, 4),
        "direction":   direction,
        "confidence":  round(confidence, 4),
    }


if __name__ == "__main__":
    # Smoke test — verify model forward pass works
    model = LSTMPredictor(input_size=25, hidden_size=128, num_layers=2, dropout=0.2)
    print("Model architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    # Test forward pass with random data
    batch_size, seq_len, n_features = 8, 60, 25
    x = torch.randn(batch_size, seq_len, n_features)
    out = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output values (should be 0-1): {out.detach().squeeze().tolist()}")
