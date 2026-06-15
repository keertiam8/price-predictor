"""
test.py — run inference on a specific stock using the trained two-stage model.

Usage:
    python src/test.py --symbol RELIANCE
    python src/test.py --symbol HDFCBANK --start 2023-01-01 --end 2024-01-01
    python src/test.py --symbol TCS --all
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "models/best_lstm_attention.pt"
DATA_PATH  = "data/combined_features.parquet"
CACHE_META = "data/cache/meta.pkl"
HORIZONS   = [5, 10, 20]
LOOKBACK   = 60
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROP_COLS  = ["date", "company_name", "industry"]

VALID_SYMBOLS = [
    "BAJFINANCE", "BHARTIARTL", "HDFCBANK", "HINDUNILVR",
    "ICICIBANK",  "LICI",       "LT",       "RELIANCE",
    "SBIN",       "TCS",
]

_DEMERGER_DATES = {
    ("BAJFINANCE", "2024-07-08"),
    ("HDFCBANK",   "2024-07-08"),
    ("RELIANCE",   "2024-07-08"),
}


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        e = self.score(lstm_out).squeeze(-1)
        a = torch.softmax(e, dim=1).unsqueeze(-1)
        return (a * lstm_out).sum(dim=1), a.squeeze(-1)


class TwoStageModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_horizons):
        super().__init__()
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0.0)
        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.cls_head  = nn.Linear(hidden_size, num_horizons)
        self.reg_head  = nn.Linear(hidden_size + num_horizons, num_horizons)

    def forward(self, x):
        lstm_out, _      = self.lstm(x)
        context, weights = self.attention(lstm_out)
        z = self.dropout(context)
        cls_logits = self.cls_head(z)
        cls_probs  = torch.sigmoid(cls_logits).detach()
        mag_preds  = F.relu(self.reg_head(torch.cat([z, cls_probs], dim=-1)))
        return cls_logits, mag_preds, weights


def _transform_to_returns(df):
    """Mirror of train.py _transform_to_returns — must stay in sync."""
    df = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)

    for sym_id, idx in df.groupby("symbol").groups.items():
        g          = df.loc[idx].sort_values("date")
        close      = g["close"]
        prev_close = close.shift(1)

        df.loc[g.index, "log_return"]  = np.log((close / prev_close).clip(1e-9))
        df.loc[g.index, "open_return"] = np.log((g["open"] / prev_close).clip(1e-9))
        df.loc[g.index, "high_ret"]    = np.log((g["high"] / close).clip(1e-9))
        df.loc[g.index, "low_ret"]     = np.log((g["low"]  / close).clip(1e-9))
        df.loc[g.index, "volume_chg"]  = g["volume"].pct_change(fill_method=None).clip(-10, 10)

        sym_str    = df.loc[g.index[0], "symbol"] if isinstance(sym_id, int) else str(sym_id)
        split_mask = g.get("is_split", pd.Series(0, index=g.index)).astype(bool)
        df.loc[g.index[split_mask.values], "log_return"]  = 0.0
        df.loc[g.index[split_mask.values], "open_return"] = 0.0
        for date_str in [d for s, d in _DEMERGER_DATES if s == sym_str]:
            dmask = g["date"].astype(str).str[:10] == date_str
            df.loc[g.index[dmask.values], "log_return"]  = 0.0
            df.loc[g.index[dmask.values], "open_return"] = 0.0

        for ma_col, new_col in [("50d_ma", "ma50_dev"), ("200d_ma", "ma200_dev")]:
            if ma_col in df.columns:
                df.loc[g.index, new_col] = (close / g[ma_col].replace(0, np.nan) - 1).clip(-2, 2)
        if "20d_avg_volume" in df.columns:
            df.loc[g.index, "vol_ratio_dev"] = (
                g["volume"] / g["20d_avg_volume"].replace(0, np.nan) - 1
            ).clip(-10, 10)

    for col in ["nifty50", "usd_inr", "brent_crude_usd", "wti_crude_usd"]:
        if col in df.columns:
            df[f"{col}_chg"] = df.groupby("symbol")[col].pct_change(fill_method=None).clip(-2, 2)

    drop_orig = ["open", "high", "low", "close", "volume",
                 "50d_ma", "200d_ma", "20d_avg_volume",
                 "nifty50", "usd_inr", "brent_crude_usd", "wti_crude_usd"]
    df = df.drop(columns=[c for c in drop_orig if c in df.columns], errors="ignore")
    return df


def run_test(symbol, start=None, end=None, show_all=False):
    if not show_all and start is None and os.path.exists(CACHE_META):
        with open(CACHE_META, "rb") as f:
            meta = pickle.load(f)
        start = meta.get("test_start_date")
        end   = end or meta.get("test_end_date")
        print(f"  Defaulting to test period: {start} -> {end}")

    print(f"\nLoading model from {MODEL_PATH} ...")
    ckpt           = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    symbol_scalers = ckpt["symbol_scalers"]
    target_scalers = ckpt.get("target_scalers", {})
    feature_cols   = ckpt["feature_cols"]
    input_size     = ckpt["feature_cols_count"]
    hidden_size    = ckpt.get("hidden_size", 256)
    num_layers     = ckpt.get("num_layers",  3)
    dropout        = ckpt.get("dropout",     0.2)

    model = TwoStageModel(input_size, hidden_size, num_layers, dropout, len(HORIZONS)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Device: {DEVICE}")

    zt_np = np.array([float(target_scalers[h].transform([[0.0]])[0, 0]) for h in HORIZONS])

    print(f"Loading data for {symbol} ...")
    raw_df = pd.read_parquet(DATA_PATH)
    raw_df = raw_df.sort_values(["symbol", "date"]).reset_index(drop=True)

    for col in ["symbol", "sector"]:
        le = LabelEncoder()
        le.fit(raw_df[col].astype(str).unique())
        raw_df[col] = le.transform(raw_df[col].astype(str))

    le_sym = LabelEncoder().fit(pd.read_parquet(DATA_PATH)["symbol"].astype(str).unique())
    sym_id = int(le_sym.transform([symbol])[0])

    if sym_id not in symbol_scalers:
        print(f"Symbol {symbol} (id={sym_id}) not found in model scalers.")
        return

    feat_sc = symbol_scalers[sym_id]["feat"]
    df      = _transform_to_returns(raw_df)
    sym_df  = df[df["symbol"] == sym_id].copy().reset_index(drop=True)

    if start:
        start_ts  = pd.Timestamp(start)
        pre_start = sym_df[sym_df["date"] < start_ts].tail(LOOKBACK)
        post      = sym_df[sym_df["date"] >= start_ts]
        if end:
            post = post[post["date"] <= pd.Timestamp(end)]
        sym_df = pd.concat([pre_start, post]).reset_index(drop=True)
    elif end:
        sym_df = sym_df[sym_df["date"] <= pd.Timestamp(end)].reset_index(drop=True)

    if len(sym_df) < LOOKBACK:
        print(f"Not enough data: need {LOOKBACK} rows, got {len(sym_df)}")
        return

    numeric_cols = sym_df.select_dtypes(include="number").columns.tolist()
    sym_df[numeric_cols] = sym_df[numeric_cols].ffill().bfill()

    available_features = [c for c in feature_cols if c in sym_df.columns]
    if len(available_features) < len(feature_cols):
        missing = set(feature_cols) - set(available_features)
        print(f"  Warning: {len(missing)} feature(s) missing: {missing}")

    feat_vals = sym_df[available_features].values.astype(np.float32)
    feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=0.0, neginf=0.0)
    feat_vals = feat_sc.transform(feat_vals)

    sym_raw = raw_df[raw_df["symbol"] == sym_id][["date", "close"]].set_index("date")["close"]

    sequences, seq_dates, raw_closes = [], [], []
    future_raw_closes = {h: [] for h in HORIZONS}
    n = len(feat_vals)

    for i in range(LOOKBACK, n):
        date = sym_df["date"].iloc[i]
        sequences.append(feat_vals[i - LOOKBACK : i])
        seq_dates.append(date)
        raw_closes.append(sym_raw.get(date, np.nan))
        for h in HORIZONS:
            if i + h < n:
                future_raw_closes[h].append(sym_raw.get(sym_df["date"].iloc[i + h], np.nan))
            else:
                future_raw_closes[h].append(np.nan)

    if not sequences:
        print("No sequences could be built.")
        return

    if start:
        start_ts = pd.Timestamp(start)
        mask     = [d >= start_ts for d in seq_dates]
        sequences         = [s for s, m in zip(sequences, mask) if m]
        seq_dates         = [d for d, m in zip(seq_dates, mask) if m]
        raw_closes        = [c for c, m in zip(raw_closes, mask) if m]
        future_raw_closes = {h: [v for v, m in zip(future_raw_closes[h], mask) if m]
                             for h in HORIZONS}

    print(f"  {len(sequences)} predictions | "
          f"{str(seq_dates[0])[:10]} -> {str(seq_dates[-1])[:10]}")

    X = torch.tensor(np.array(sequences, dtype=np.float32)).to(DEVICE)
    with torch.no_grad():
        cls_logits, mag_preds, _ = model(X)

    cls_np  = cls_logits.cpu().numpy()  # (N, H) direction logits
    mag_np  = mag_preds.cpu().numpy()   # (N, H) magnitude predictions
    pred_up = cls_np > 0                # (N, H) UP/DOWN flags

    # Reconstruct predicted log-returns: sign(cls) × magnitude, then invert scaler
    preds_return = np.zeros_like(cls_np)
    for i, h in enumerate(HORIZONS):
        sign              = np.where(pred_up[:, i], 1.0, -1.0)
        final_std         = zt_np[i] + sign * mag_np[:, i]
        preds_return[:, i] = target_scalers[h].inverse_transform(
            final_std.reshape(-1, 1)
        ).ravel()

    rc = np.array(raw_closes)

    # ── Prediction table ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  PREDICTIONS FOR {symbol}"
          + (f"  |  {start} -> {end}" if start or end else ""))
    print(f"  (values show predicted % change from current close)")
    print(f"{'='*80}")
    print(f"  {'Date':>12}  {'Close':>9}  {'Pred 5d':>10}  {'Pred 10d':>10}"
          f"  {'Pred 20d':>10}  {'Dir(5d)':>8}")
    print(f"  {'─'*76}")

    show = min(len(seq_dates), 30)
    for i in range(show):
        date = str(seq_dates[i])[:10]
        cl   = rc[i]
        r5, r10, r20 = preds_return[i]
        arrow = "UP" if pred_up[i, 0] else "DN"
        print(f"  {date:>12}  {cl:>9.2f}  {cl*np.exp(r5):>10.2f}"
              f"  {cl*np.exp(r10):>10.2f}  {cl*np.exp(r20):>10.2f}  {arrow:>8}")

    if len(seq_dates) > 30:
        print(f"  ... showing 30 of {len(seq_dates)} predictions")

    # ── Accuracy summary ───────────────────────────────────────────────────
    print(f"\n  ACCURACY SUMMARY  ({len(sequences)} sequences)")
    print(f"  {'─'*66}")
    print(f"  {'Horizon':>10}  {'Dir Acc':>9}  {'RMSE(ret)':>10}  {'MAE(ret)':>9}"
          f"  {'Prec':>7}  {'Rec':>7}  {'Correct/Total':>14}")
    print(f"  {'─'*66}")

    for h_idx, h in enumerate(HORIZONS):
        fut   = np.array(future_raw_closes[h])
        valid = ~np.isnan(fut) & ~np.isnan(rc)
        if valid.sum() == 0:
            continue

        actual_log_ret = np.log((fut[valid] / rc[valid]).clip(1e-9))
        pred_log_ret   = preds_return[valid, h_idx]
        pred_up_h      = pred_up[valid, h_idx]
        actual_up      = actual_log_ret > 0

        n_correct = (pred_up_h == actual_up).sum()
        dir_acc   = 100.0 * n_correct / valid.sum()
        rmse      = np.sqrt(np.nanmean((pred_log_ret - actual_log_ret) ** 2))
        mae       = np.nanmean(np.abs(pred_log_ret - actual_log_ret))

        from sklearn.metrics import precision_score, recall_score
        prec = precision_score(actual_up.astype(int), pred_up_h.astype(int), zero_division=0) * 100
        rec  = recall_score(actual_up.astype(int), pred_up_h.astype(int), zero_division=0) * 100

        print(f"  {h:>8d}d  {dir_acc:>8.2f}%  {rmse:>10.4f}  {mae:>9.4f}"
              f"  {prec:>6.1f}%  {rec:>6.1f}%  {n_correct:>6}/{valid.sum():<7}")

    print(f"  {'─'*66}")

    # ── Latest prediction ──────────────────────────────────────────────────
    last_close = rc[-1]
    print(f"\n  Latest prediction (as of {str(seq_dates[-1])[:10]}):")
    print(f"    Current close  : {last_close:>10.2f}")
    for h_idx, h in enumerate(HORIZONS):
        r     = preds_return[-1, h_idx]
        pct   = (np.exp(r) - 1) * 100
        arrow = "UP  ^" if pred_up[-1, h_idx] else "DOWN v"
        print(f"    In {h:2d} days      :  {pct:>+7.2f}%  "
              f"(implied ~{last_close * np.exp(r):.2f})  {arrow}")

    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, choices=VALID_SYMBOLS)
    parser.add_argument("--start",  default=None)
    parser.add_argument("--end",    default=None)
    parser.add_argument("--all",    action="store_true")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"No trained model at {MODEL_PATH} — run train.py first.")
        sys.exit(1)

    run_test(args.symbol, args.start, args.end, show_all=args.all)
