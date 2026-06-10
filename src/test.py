"""
test.py — run inference on a specific stock using the trained model.

Usage:
    python src/test.py --symbol RELIANCE
    python src/test.py --symbol HDFCBANK --start 2023-01-01 --end 2024-01-01
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

MODEL_PATH  = "models/best_lstm_attention.pt"
DATA_PATH   = "data/combined_features.parquet"
CACHE_META  = "data/cache/meta.pkl"
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


def run_test(symbol, start=None, end=None):
    # Default to test period so we never accidentally show training/val predictions
    if start is None and os.path.exists(CACHE_META):
        with open(CACHE_META, "rb") as f:
            meta = pickle.load(f)
        start = meta.get("test_start_date")
        end   = end or meta.get("test_end_date")
        print(f"  Defaulting to test period: {start} -> {end}")

    print(f"\nLoading model from {MODEL_PATH} ...")
    checkpoint     = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    symbol_scalers = checkpoint["symbol_scalers"]   # {sym_id: {"feat": sc, "tgt": sc}}
    feature_cols   = checkpoint["feature_cols"]
    close_col_idx  = checkpoint["close_col_idx"]
    input_size     = checkpoint["feature_cols_count"]

    model = LSTMAttentionModel(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, len(HORIZONS)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"  Device: {DEVICE}")

    print(f"Loading data for {symbol} ...")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Label encode exactly as in training
    for col in ["symbol", "sector", "cap_category"]:
        le = LabelEncoder()
        le.fit(df[col].astype(str).unique())
        df[col] = le.transform(df[col].astype(str))

    # Find numeric symbol code for the requested symbol
    le_sym = LabelEncoder().fit(pd.read_parquet(DATA_PATH)["symbol"].astype(str).unique())
    sym_id = int(le_sym.transform([symbol])[0])

    if sym_id not in symbol_scalers:
        print(f"Symbol {symbol} (id={sym_id}) not found in model scalers.")
        return

    feat_sc = symbol_scalers[sym_id]["feat"]
    tgt_sc  = symbol_scalers[sym_id]["tgt"]

    sym_df = df[df["symbol"] == sym_id].copy().reset_index(drop=True)

    # Keep LOOKBACK rows before start for context
    if start:
        start_ts   = pd.Timestamp(start)
        pre_start  = sym_df[sym_df["date"] < start_ts].tail(LOOKBACK)
        post_start = sym_df[sym_df["date"] >= start_ts]
        if end:
            post_start = post_start[post_start["date"] <= pd.Timestamp(end)]
        sym_df = pd.concat([pre_start, post_start]).reset_index(drop=True)
    elif end:
        sym_df = sym_df[sym_df["date"] <= pd.Timestamp(end)].reset_index(drop=True)

    if len(sym_df) < LOOKBACK:
        print(f"Not enough data: need {LOOKBACK} rows, got {len(sym_df)}")
        return

    # Fill and scale using this symbol's scaler
    numeric_cols = sym_df.select_dtypes(include="number").columns.tolist()
    sym_df[numeric_cols] = sym_df[numeric_cols].ffill().bfill()

    feat_vals = sym_df[feature_cols].values.astype(np.float32)
    feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=1.0, neginf=0.0)
    feat_vals = feat_sc.transform(feat_vals)        # clip=True in scaler keeps values in [0,1]
    feat_vals = np.clip(feat_vals, 0.0, 1.0)        # safety clip in case old scaler lacks clip=True

    # Build sequences — also collect actual future closes for accuracy calculation
    sequences, seq_dates, actual_closes, current_closes_scaled = [], [], [], []
    future_closes = {h: [] for h in HORIZONS}   # actual close h days ahead
    n = len(feat_vals)
    for i in range(LOOKBACK, n):
        sequences.append(feat_vals[i - LOOKBACK : i])
        seq_dates.append(sym_df["date"].iloc[i])
        actual_closes.append(sym_df["close"].iloc[i])
        current_closes_scaled.append(feat_vals[i, close_col_idx])
        for h in HORIZONS:
            future_closes[h].append(
                sym_df["close"].iloc[i + h] if i + h < n else np.nan
            )

    if not sequences:
        print("No sequences could be built for the given date range.")
        return

    print(f"  {len(sequences)} predictions | "
          f"{str(seq_dates[0])[:10]} -> {str(seq_dates[-1])[:10]}")

    X = torch.tensor(np.array(sequences, dtype=np.float32)).to(DEVICE)
    with torch.no_grad():
        preds_scaled, _ = model(X)

    preds_scaled = preds_scaled.cpu().numpy()
    preds_price  = tgt_sc.inverse_transform(preds_scaled)
    cc_scaled    = np.array(current_closes_scaled)   # shape (N,)

    # ── Print table ──────────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print(f"  PREDICTIONS FOR {symbol}"
          + (f"  |  {start} -> {end}" if start or end else ""))
    print(f"{'='*76}")
    print(f"  {'Date':>12}  {'Actual':>10}  {'5d Pred':>10}  {'10d Pred':>10}  {'20d Pred':>10}  {'Dir':>5}")
    print(f"  {'─'*64}")

    show = min(len(seq_dates), 30)
    for i in range(show):
        date      = str(seq_dates[i])[:10]
        actual    = actual_closes[i]
        p5, p10, p20 = preds_price[i]
        # direction based on 5d prediction vs current close
        arrow = "UP" if preds_scaled[i, 0] > cc_scaled[i] else "DN"
        print(f"  {date:>12}  {actual:>10.2f}  {p5:>10.2f}  {p10:>10.2f}  {p20:>10.2f}  {arrow:>5}")

    if len(seq_dates) > 30:
        print(f"  ... showing 30 of {len(seq_dates)} predictions")

    # ── Accuracy summary ─────────────────────────────────────────────────
    ac = np.array(actual_closes)
    print(f"\n  ACCURACY SUMMARY  ({len(sequences)} sequences)")
    print(f"  {'─'*58}")
    print(f"  {'Horizon':>10}  {'Dir Acc':>9}  {'RMSE(sc)':>9}  {'MAE(sc)':>8}  {'Correct/Total':>14}")
    print(f"  {'─'*58}")
    for h_idx, h in enumerate(HORIZONS):
        fut = np.array(future_closes[h])
        valid = ~np.isnan(fut)                         # last h rows have no future close
        if valid.sum() == 0:
            continue
        pred_up   = preds_scaled[valid, h_idx] > cc_scaled[valid]
        actual_up = fut[valid] > ac[valid]             # price direction in rupee space
        n_correct = (pred_up == actual_up).sum()
        dir_acc   = 100.0 * n_correct / valid.sum()

        rmse_s = np.sqrt(np.mean((preds_scaled[valid, h_idx] - cc_scaled[valid]) ** 2))
        mae_s  = np.mean(np.abs(preds_scaled[valid, h_idx] - cc_scaled[valid]))

        print(f"  {h:>8d}d  {dir_acc:>8.2f}%  {rmse_s:>9.4f}  {mae_s:>8.4f}"
              f"  {n_correct:>6}/{valid.sum():<7}")
    print(f"  {'─'*58}")

    # ── Latest prediction ────────────────────────────────────────────────
    last_actual = actual_closes[-1]
    last_preds  = preds_price[-1]
    last_cc     = cc_scaled[-1]

    print(f"\n  Latest prediction (as of {str(seq_dates[-1])[:10]}):")
    print(f"    Current close  : {last_actual:>10.2f}")
    for h_idx, h in enumerate(HORIZONS):
        p      = last_preds[h_idx]
        change = ((p - last_actual) / last_actual) * 100
        up     = preds_scaled[-1, h_idx] > last_cc
        arrow  = "UP  ^" if up else "DOWN v"
        print(f"    In {h:2d} days      : {p:>10.2f}  ({change:+.2f}%)  {arrow}")

    print(f"{'='*76}")


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
