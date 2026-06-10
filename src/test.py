"""
test.py — run inference on a specific stock and date range using the trained model.

Usage:
    python src/test.py --symbol RELIANCE
    python src/test.py --symbol HDFCBANK --start 2023-01-01 --end 2024-01-01
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

CACHE_DIR   = "data/cache"
MODEL_PATH  = "models/best_lstm_attention.pt"
DATA_PATH   = "data/combined_features.parquet"
HORIZONS    = [5, 10, 20]
LOOKBACK    = 60
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROP_COLS   = ["date", "company_name", "industry"]

VALID_SYMBOLS = [
    "BAJFINANCE", "BHARTIARTL", "HDFCBANK", "HINDUNILVR",
    "ICICIBANK",  "LICI",       "LT",       "RELIANCE",
    "SBIN",       "TCS"
]

# ── Model ─────────────────────────────────────────────────────────────────
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        e = self.score(lstm_out).squeeze(-1)
        a = torch.softmax(e, dim=1).unsqueeze(-1)
        return (a * lstm_out).sum(dim=1), a.squeeze(-1)


class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_outputs):
        super().__init__()
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0.0)
        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        lstm_out, _      = self.lstm(x)
        context, weights = self.attention(lstm_out)
        return self.fc(self.dropout(context)), weights


# ── Inference ─────────────────────────────────────────────────────────────
def run_test(symbol, start=None, end=None):
    print(f"\nLoading model from {MODEL_PATH} ...")
    checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    feat_scaler = checkpoint["feat_scaler"]
    tgt_scaler  = checkpoint["tgt_scaler"]
    input_size  = checkpoint["feature_cols_count"]

    model = LSTMAttentionModel(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, len(HORIZONS)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"  Device: {DEVICE}")

    print(f"Loading data for {symbol} ...")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Label encode same as training
    for col in ["symbol", "sector", "cap_category"]:
        le = LabelEncoder()
        le.fit(df[col].astype(str).unique())
        df[col] = le.transform(df[col].astype(str))

    feature_cols = [c for c in df.columns
                    if c not in DROP_COLS and not c.startswith("target_")]

    # Filter to requested symbol AFTER encoding
    sym_code = LabelEncoder().fit(
        pd.read_parquet(DATA_PATH)["symbol"].astype(str).unique()
    ).transform([symbol])[0]

    sym_df = df[df["symbol"] == sym_code].copy().reset_index(drop=True)

    # Apply date filter
    if start:
        # keep LOOKBACK rows before start so sequences can be built
        start_ts  = pd.Timestamp(start)
        pre_start = sym_df[sym_df["date"] < start_ts].tail(LOOKBACK)
        post_start = sym_df[sym_df["date"] >= start_ts]
        if end:
            post_start = post_start[post_start["date"] <= pd.Timestamp(end)]
        sym_df = pd.concat([pre_start, post_start]).reset_index(drop=True)
    elif end:
        sym_df = sym_df[sym_df["date"] <= pd.Timestamp(end)].reset_index(drop=True)

    if len(sym_df) < LOOKBACK:
        print(f"Not enough data: need {LOOKBACK} rows, got {len(sym_df)}")
        return

    # Fill then scale
    numeric_cols = sym_df.select_dtypes(include="number").columns.tolist()
    sym_df[numeric_cols] = sym_df[numeric_cols].ffill().bfill()

    feat_vals = sym_df[feature_cols].values.astype(np.float32)
    feat_vals = feat_scaler.transform(feat_vals)
    feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=1.0, neginf=0.0)

    # Build sequences
    sequences, seq_dates, actual_closes = [], [], []
    for i in range(LOOKBACK, len(feat_vals)):
        sequences.append(feat_vals[i - LOOKBACK : i])
        seq_dates.append(sym_df["date"].iloc[i])
        actual_closes.append(sym_df["close"].iloc[i])

    if not sequences:
        print("No sequences could be built for the given date range.")
        return

    print(f"  {len(sequences)} predictions | "
          f"{str(seq_dates[0])[:10]} -> {str(seq_dates[-1])[:10]}")

    X = torch.tensor(np.array(sequences, dtype=np.float32)).to(DEVICE)
    with torch.no_grad():
        preds_norm, _ = model(X)

    preds_price = tgt_scaler.inverse_transform(preds_norm.cpu().numpy())

    # ── Print results ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  PREDICTIONS FOR {symbol}"
          + (f"  |  {start} -> {end}" if start or end else ""))
    print(f"{'='*72}")
    print(f"  {'Date':>12}  {'Actual':>10}  {'5d Pred':>10}  {'10d Pred':>10}  {'20d Pred':>10}")
    print(f"  {'─'*60}")

    show = min(len(seq_dates), 30)
    for i in range(show):
        date   = str(seq_dates[i])[:10]
        actual = actual_closes[i]
        print(f"  {date:>12}  {actual:>10.2f}"
              f"  {preds_price[i,0]:>10.2f}"
              f"  {preds_price[i,1]:>10.2f}"
              f"  {preds_price[i,2]:>10.2f}")

    if len(seq_dates) > 30:
        print(f"  ... showing 30 of {len(seq_dates)} predictions")

    # ── Latest prediction summary ────────────────────────────────────────
    last_date   = str(seq_dates[-1])[:10]
    last_actual = actual_closes[-1]
    print(f"\n  Latest prediction  (as of {last_date}):")
    print(f"    Current close  : {last_actual:>10.2f}")
    for h, p in zip(HORIZONS, preds_price[-1]):
        change = ((p - last_actual) / last_actual) * 100
        arrow  = "UP  ^" if p > last_actual else "DOWN v"
        print(f"    In {h:2d} days      : {p:>10.2f}  ({change:+.2f}%)  {arrow}")

    print(f"{'='*72}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, choices=VALID_SYMBOLS,
                        help=f"One of: {', '.join(VALID_SYMBOLS)}")
    parser.add_argument("--start",  default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=None, help="End date   YYYY-MM-DD")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"No trained model at {MODEL_PATH} — run train.py first.")
        sys.exit(1)

    run_test(args.symbol, args.start, args.end)
