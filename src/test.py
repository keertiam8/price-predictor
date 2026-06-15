"""
test.py — run inference on a specific stock using the trained model.

Predictions are expressed as log-returns (% change from current close).
Features are converted to the same return/ratio representation used in training.

Usage:
    python src/test.py --symbol RELIANCE
    python src/test.py --symbol HDFCBANK --start 2023-01-01 --end 2024-01-01
    python src/test.py --symbol TCS --all      # show all history, not just test period
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


class TanhGateLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight_ih = nn.Linear(input_size,  4 * hidden_size, bias=True)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x, state):
        h, c = state
        gates = self.weight_ih(x) + self.weight_hh(h)
        i, f, g, o = gates.chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.tanh(o) * torch.tanh(c_new)
        return h_new, c_new


class TanhGateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        sizes = [input_size] + [hidden_size] * num_layers
        self.cells = nn.ModuleList(TanhGateLSTMCell(sizes[i], hidden_size) for i in range(num_layers))
        self.drop  = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None

    def forward(self, x):
        B, T, _ = x.shape
        dev = x.device
        h = [torch.zeros(B, self.hidden_size, device=dev) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_size, device=dev) for _ in range(self.num_layers)]
        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for layer, cell in enumerate(self.cells):
                h[layer], c[layer] = cell(inp, (h[layer], c[layer]))
                inp = h[layer]
                if self.drop and layer < self.num_layers - 1:
                    inp = self.drop(inp)
            outputs.append(h[-1])
        return torch.stack(outputs, dim=1), (torch.stack(h, dim=0), torch.stack(c, dim=0))


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        e = self.score(lstm_out).squeeze(-1)
        a = torch.softmax(e, dim=1).unsqueeze(-1)
        return (a * lstm_out).sum(dim=1), a.squeeze(-1)


class LSTMAttentionModel(nn.Module):
    """Dual-head LSTM+attention (must match train.py).
    Uses TanhGateLSTM (tanh output gate) — must stay in sync with train.py."""
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_outputs):
        super().__init__()
        self.lstm      = TanhGateLSTM(input_size, hidden_size, num_layers,
                                      dropout=dropout if num_layers > 1 else 0.0)
        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.reg_head  = nn.Linear(hidden_size, num_outputs)
        self.cls_head  = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        lstm_out, _      = self.lstm(x)
        context, weights = self.attention(lstm_out)
        z = self.dropout(context)
        return self.reg_head(z), self.cls_head(z), weights


def _transform_to_returns(df):
    """Mirror of train.py's _transform_to_returns — must stay in sync."""
    df = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)

    for _, idx in df.groupby("symbol").groups.items():
        g = df.loc[idx].sort_values("date")
        close      = g["close"]
        prev_close = close.shift(1)

        df.loc[g.index, "log_return"]  = np.log((close / prev_close).clip(1e-9))
        df.loc[g.index, "open_return"] = np.log((g["open"] / prev_close).clip(1e-9))
        df.loc[g.index, "high_ret"]    = np.log((g["high"] / close).clip(1e-9))
        df.loc[g.index, "low_ret"]     = np.log((g["low"]  / close).clip(1e-9))
        df.loc[g.index, "volume_chg"]  = g["volume"].pct_change(fill_method=None).clip(-10, 10)

        for ma_col, new_col in [("50d_ma", "ma50_dev"), ("200d_ma", "ma200_dev")]:
            if ma_col in df.columns:
                df.loc[g.index, new_col] = (close / g[ma_col].replace(0, np.nan) - 1).clip(-2, 2)
        if "20d_avg_volume" in df.columns:
            df.loc[g.index, "vol_ratio_dev"] = (
                g["volume"] / g["20d_avg_volume"].replace(0, np.nan) - 1
            ).clip(-10, 10)

        mcap = g["avg_mcap_cr"].replace(0, np.nan) if "avg_mcap_cr" in df.columns else None
        for col in ["revenue", "net_profit", "ebitda", "assets", "equity", "debt",
                    "operating_cash_flow", "free_cash_flow"]:
            if col in df.columns and mcap is not None:
                df.loc[g.index, f"{col}_to_mcap"] = (g[col] / mcap).clip(-100, 100)
        if "avg_mcap_cr" in df.columns:
            df.loc[g.index, "mcap_chg"] = g["avg_mcap_cr"].pct_change(fill_method=None).clip(-2, 2)

    for col in ["bse_sensex", "nifty50", "gold_inr", "gold_usd",
                "brent_crude_usd", "wti_crude_usd", "usd_inr",
                "us_cpi_index", "us_gdp_usd_bn", "india_gdp_usd_bn"]:
        if col in df.columns:
            df[f"{col}_chg"] = df.groupby("symbol")[col].pct_change(fill_method=None).clip(-2, 2)

    drop_orig = (
        ["open", "high", "low", "close", "volume", "50d_ma", "200d_ma", "20d_avg_volume",
         "avg_mcap_cr", "revenue", "net_profit", "ebitda", "assets", "equity", "debt",
         "operating_cash_flow", "free_cash_flow"]
        + ["bse_sensex", "nifty50", "gold_inr", "gold_usd", "brent_crude_usd",
           "wti_crude_usd", "usd_inr", "us_cpi_index", "us_gdp_usd_bn", "india_gdp_usd_bn"]
    )
    df = df.drop(columns=[c for c in drop_orig if c in df.columns], errors="ignore")
    return df


def run_test(symbol, start=None, end=None, show_all=False):
    # Default to test period unless --all is passed
    if not show_all and start is None and os.path.exists(CACHE_META):
        with open(CACHE_META, "rb") as f:
            meta = pickle.load(f)
        start = meta.get("test_start_date")
        end   = end or meta.get("test_end_date")
        print(f"  Defaulting to test period: {start} -> {end}")

    print(f"\nLoading model from {MODEL_PATH} ...")
    checkpoint     = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    symbol_scalers = checkpoint["symbol_scalers"]
    target_scalers = checkpoint.get("target_scalers", {})
    feature_cols   = checkpoint["feature_cols"]
    input_size  = checkpoint["feature_cols_count"]
    hidden_size = checkpoint.get("hidden_size", HIDDEN_SIZE)
    num_layers  = checkpoint.get("num_layers",  NUM_LAYERS)
    dropout     = checkpoint.get("dropout",     DROPOUT)

    model = LSTMAttentionModel(input_size, hidden_size, num_layers, dropout, len(HORIZONS)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"  Device: {DEVICE}")

    print(f"Loading data for {symbol} ...")
    raw_df = pd.read_parquet(DATA_PATH)
    raw_df = raw_df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Label encode
    for col in ["symbol", "sector", "cap_category"]:
        le = LabelEncoder()
        le.fit(raw_df[col].astype(str).unique())
        raw_df[col] = le.transform(raw_df[col].astype(str))

    le_sym = LabelEncoder().fit(pd.read_parquet(DATA_PATH)["symbol"].astype(str).unique())
    sym_id = int(le_sym.transform([symbol])[0])

    if sym_id not in symbol_scalers:
        print(f"Symbol {symbol} (id={sym_id}) not found in model scalers.")
        return

    feat_sc = symbol_scalers[sym_id]["feat"]

    # Apply return transformations (same as training pipeline)
    df = _transform_to_returns(raw_df)

    sym_df = df[df["symbol"] == sym_id].copy().reset_index(drop=True)

    # Keep LOOKBACK rows before start for context window
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

    # Fill and scale
    numeric_cols = sym_df.select_dtypes(include="number").columns.tolist()
    sym_df[numeric_cols] = sym_df[numeric_cols].ffill().bfill()

    # Only keep feature_cols that exist (handles version mismatches)
    available_features = [c for c in feature_cols if c in sym_df.columns]
    if len(available_features) < len(feature_cols):
        missing = set(feature_cols) - set(available_features)
        print(f"  Warning: {len(missing)} feature(s) missing after transform: {missing}")

    feat_vals = sym_df[available_features].values.astype(np.float32)
    feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=0.0, neginf=0.0)
    feat_vals = feat_sc.transform(feat_vals)

    # Build sequences
    n = len(feat_vals)
    sequences, seq_dates, raw_closes = [], [], []
    future_raw_closes = {h: [] for h in HORIZONS}

    # Rebuild raw close from original data for this symbol
    sym_raw = raw_df[raw_df["symbol"] == sym_id][["date", "close"]].set_index("date")["close"]

    for i in range(LOOKBACK, n):
        date = sym_df["date"].iloc[i]
        sequences.append(feat_vals[i - LOOKBACK : i])
        seq_dates.append(date)
        raw_closes.append(sym_raw.get(date, np.nan))

        for h in HORIZONS:
            if i + h < n:
                future_date = sym_df["date"].iloc[i + h]
                future_raw_closes[h].append(sym_raw.get(future_date, np.nan))
            else:
                future_raw_closes[h].append(np.nan)

    if not sequences:
        print("No sequences could be built for the given date range.")
        return

    # Filter to only rows within the requested date window (exclude pre-start context rows)
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
        reg_scaled, cls_logits, _ = model(X)

    preds_std = reg_scaled.cpu().numpy()     # regression head (N, 3), standardised
    cls_logit = cls_logits.cpu().numpy()     # classification head (N, 3), direction logits
    pred_up   = cls_logit > 0                # True ⇔ classifier predicts UP

    # Invert StandardScaler to get raw log-returns (magnitude from regression head)
    preds_return = np.stack([
        target_scalers[h].inverse_transform(preds_std[:, i:i+1]).ravel()
        if h in target_scalers else preds_std[:, i]
        for i, h in enumerate(HORIZONS)
    ], axis=1)
    preds_pct = (np.exp(preds_return) - 1) * 100   # convert to % change

    rc = np.array(raw_closes)

    # ── Print table ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  PREDICTIONS FOR {symbol}"
          + (f"  |  {start} -> {end}" if start or end else ""))
    print(f"  (values show predicted % change from current close)")
    print(f"{'='*80}")
    print(f"  {'Date':>12}  {'Close':>9}  {'Pred 5d':>10}  {'Pred 10d':>10}  {'Pred 20d':>10}  {'Dir(5d)':>8}")
    print(f"  {'─'*76}")

    show = min(len(seq_dates), 30)
    for i in range(show):
        date = str(seq_dates[i])[:10]
        cl   = rc[i]
        r5, r10, r20 = preds_return[i]
        arrow = "UP" if pred_up[i, 0] else "DN"
        p5   = cl * np.exp(r5)
        p10  = cl * np.exp(r10)
        p20  = cl * np.exp(r20)
        print(f"  {date:>12}  {cl:>9.2f}  {p5:>10.2f}  {p10:>10.2f}  {p20:>10.2f}  {arrow:>8}")

    if len(seq_dates) > 30:
        print(f"  ... showing 30 of {len(seq_dates)} predictions")

    # ── Accuracy summary ─────────────────────────────────────────────────
    print(f"\n  ACCURACY SUMMARY  ({len(sequences)} sequences)")
    print(f"  {'─'*62}")
    print(f"  {'Horizon':>10}  {'Dir Acc':>9}  {'RMSE(ret)':>10}  {'MAE(ret)':>9}  {'Correct/Total':>14}")
    print(f"  {'─'*62}")
    for h_idx, h in enumerate(HORIZONS):
        fut  = np.array(future_raw_closes[h])
        valid = ~np.isnan(fut) & ~np.isnan(rc)
        if valid.sum() == 0:
            continue

        # Actual log-return for the h-day window
        actual_log_ret = np.log((fut[valid] / rc[valid]).clip(1e-9))

        # Predicted log-return (inverse-transformed)
        pred_log_ret = preds_return[valid, h_idx]

        pred_up_h = pred_up[valid, h_idx]             # classifier says UP
        actual_up = actual_log_ret > 0                # actual was UP
        n_correct = (pred_up_h == actual_up).sum()
        dir_acc   = 100.0 * n_correct / valid.sum()

        rmse = np.sqrt(np.nanmean((pred_log_ret - actual_log_ret) ** 2))
        mae  = np.nanmean(np.abs(pred_log_ret - actual_log_ret))

        print(f"  {h:>8d}d  {dir_acc:>8.2f}%  {rmse:>10.4f}  {mae:>9.4f}"
              f"  {n_correct:>6}/{valid.sum():<7}")
    print(f"  {'─'*62}")

    # ── Latest prediction ────────────────────────────────────────────────
    last_close = rc[-1]
    last_pct   = preds_pct[-1]

    print(f"\n  Latest prediction (as of {str(seq_dates[-1])[:10]}):")
    print(f"    Current close  : {last_close:>10.2f}")
    for h_idx, h in enumerate(HORIZONS):
        pct   = last_pct[h_idx]
        arrow = "UP  ^" if pred_up[-1, h_idx] else "DOWN v"
        implied_price = last_close * np.exp(preds_return[-1, h_idx])
        print(f"    In {h:2d} days      :  {pct:>+7.2f}%  (implied ~{implied_price:.2f})  {arrow}")

    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, choices=VALID_SYMBOLS)
    parser.add_argument("--start",  default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=None, help="End date   YYYY-MM-DD")
    parser.add_argument("--all",    action="store_true",
                        help="Show full history, not just test period")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"No trained model at {MODEL_PATH} — run train.py first.")
        sys.exit(1)

    run_test(args.symbol, args.start, args.end, show_all=args.all)
