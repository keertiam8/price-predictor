"""
test_timesfm.py — run TimesFM inference on a specific stock.

Mirrors test.py output format exactly. No trained model needed —
TimesFM is used zero-shot directly from the pre-trained checkpoint.

Usage:
    python src/test_timesfm.py --symbol RELIANCE
    python src/test_timesfm.py --symbol HDFCBANK --start 2023-01-01 --end 2024-01-01
    python src/test_timesfm.py --symbol TCS --all
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

import timesfm
try:
    from importlib.metadata import version as pkg_version
    TIMESFM_VERSION = pkg_version('timesfm')
except Exception:
    TIMESFM_VERSION = 'unknown'

DATA_PATH    = "data/combined_features.parquet"
CACHE_META   = "data/cache_timesfm/meta.pkl"
HORIZONS     = [5, 10, 20]
HORIZON_LEN  = max(HORIZONS)
LOOKBACK     = 60
BATCH_SIZE   = 128
DEVICE_STR   = "gpu" if torch.cuda.is_available() else "cpu"
TIMESFM_REPO = "google/timesfm-2.0-200m-pytorch"

VALID_SYMBOLS = [
    "BAJFINANCE", "BHARTIARTL", "HDFCBANK", "HINDUNILVR",
    "ICICIBANK",  "LICI",       "LT",       "RELIANCE",
    "SBIN",       "TCS"
]

_OHLCV_PRICE = ["open", "high", "low", "close", "volume"]
_MA_COLS     = ["50d_ma", "200d_ma", "20d_avg_volume"]
_MACRO_LEVEL = ["bse_sensex", "nifty50", "gold_inr", "gold_usd",
                "brent_crude_usd", "wti_crude_usd", "usd_inr",
                "avg_mcap_cr", "us_cpi_index", "us_gdp_usd_bn", "india_gdp_usd_bn"]


def _transform_to_returns(df):
    """Mirror of train.py — must stay in sync."""
    df = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    for _, idx in df.groupby("symbol").groups.items():
        g = df.loc[idx].sort_values("date")
        close      = g["close"]
        prev_close = close.shift(1)

        df.loc[g.index, "log_return"]   = np.log((close / prev_close).clip(1e-9))
        df.loc[g.index, "open_return"]  = np.log((g["open"] / prev_close).clip(1e-9))
        df.loc[g.index, "high_ret"]     = np.log((g["high"] / close).clip(1e-9))
        df.loc[g.index, "low_ret"]      = np.log((g["low"]  / close).clip(1e-9))
        df.loc[g.index, "volume_chg"]   = g["volume"].pct_change(fill_method=None).clip(-10, 10)

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
        _OHLCV_PRICE + _MA_COLS
        + ["avg_mcap_cr", "revenue", "net_profit", "ebitda", "assets",
           "equity", "debt", "operating_cash_flow", "free_cash_flow"]
        + [c for c in _MACRO_LEVEL if c != "avg_mcap_cr"]
    )
    df = df.drop(columns=[c for c in drop_orig if c in df.columns], errors="ignore")
    return df


def run_test(symbol, start=None, end=None, show_all=False):
    if not show_all and start is None and os.path.exists(CACHE_META):
        with open(CACHE_META, "rb") as f:
            meta = pickle.load(f)
        start = meta.get("test_start_date")
        end   = end or meta.get("test_end_date")
        print(f"  Defaulting to test period: {start} -> {end}")

    print(f"\nLoading TimesFM from {TIMESFM_REPO}  (device: {DEVICE_STR}) ...")
    config = timesfm.ForecastConfig(
        max_context=LOOKBACK,
        max_horizon=HORIZON_LEN,
        per_core_batch_size=BATCH_SIZE,
    )
    tfm = timesfm.TimesFM_2p5_200M_torch(config=config)
    tfm.load_checkpoint(repo_id=TIMESFM_REPO)
    print("  TimesFM loaded.")

    print(f"Loading data for {symbol} ...")
    raw_df = pd.read_parquet(DATA_PATH)
    raw_df = raw_df.sort_values(["symbol", "date"]).reset_index(drop=True)

    for col in ["symbol", "sector", "cap_category"]:
        le = LabelEncoder()
        le.fit(raw_df[col].astype(str).unique())
        raw_df[col] = le.transform(raw_df[col].astype(str))

    le_sym = LabelEncoder().fit(pd.read_parquet(DATA_PATH)["symbol"].astype(str).unique())
    sym_id = int(le_sym.transform([symbol])[0])

    df = _transform_to_returns(raw_df)
    sym_df = df[df["symbol"] == sym_id].copy().reset_index(drop=True)

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

    sym_raw = raw_df[raw_df["symbol"] == sym_id][["date", "close"]].set_index("date")["close"]
    lr_vals = sym_df["log_return"].values.astype(np.float32)

    sequences, seq_dates, raw_closes = [], [], []
    future_raw_closes = {h: [] for h in HORIZONS}
    n = len(sym_df)

    for i in range(LOOKBACK, n):
        date = sym_df["date"].iloc[i]
        sequences.append(lr_vals[i - LOOKBACK : i])
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

    if start:
        start_ts = pd.Timestamp(start)
        mask = [d >= start_ts for d in seq_dates]
        sequences         = [s for s, m in zip(sequences, mask) if m]
        seq_dates         = [d for d, m in zip(seq_dates, mask) if m]
        raw_closes        = [c for c, m in zip(raw_closes, mask) if m]
        future_raw_closes = {h: [v for v, m in zip(future_raw_closes[h], mask) if m]
                             for h in HORIZONS}

    print(f"  {len(sequences)} predictions | "
          f"{str(seq_dates[0])[:10]} -> {str(seq_dates[-1])[:10]}")

    # Batched TimesFM inference
    all_forecasts = []
    for start_idx in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[start_idx : start_idx + BATCH_SIZE]
        point_forecast, _ = tfm.forecast(batch, freq=[0] * len(batch), horizon_len=HORIZON_LEN)
        all_forecasts.append(point_forecast)

    forecasts = np.concatenate(all_forecasts, axis=0)   # (N, HORIZON_LEN)

    # Cumulative sum → multi-step log-return predictions
    preds_return = np.stack([forecasts[:, :h].sum(axis=1) for h in HORIZONS], axis=1)
    preds_pct    = (np.exp(preds_return) - 1) * 100

    rc = np.array(raw_closes)

    # ── Print table ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  PREDICTIONS FOR {symbol} [TimesFM]"
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
        arrow = "UP" if r5 > 0 else "DN"
        p5  = cl * np.exp(r5)
        p10 = cl * np.exp(r10)
        p20 = cl * np.exp(r20)
        print(f"  {date:>12}  {cl:>9.2f}  {p5:>10.2f}  {p10:>10.2f}  {p20:>10.2f}  {arrow:>8}")

    if len(seq_dates) > 30:
        print(f"  ... showing 30 of {len(seq_dates)} predictions")

    # ── Accuracy summary ─────────────────────────────────────────────────
    print(f"\n  ACCURACY SUMMARY  ({len(sequences)} sequences)")
    print(f"  {'─'*62}")
    print(f"  {'Horizon':>10}  {'Dir Acc':>9}  {'RMSE(ret)':>10}  {'MAE(ret)':>9}  {'Correct/Total':>14}")
    print(f"  {'─'*62}")
    for h_idx, h in enumerate(HORIZONS):
        fut   = np.array(future_raw_closes[h])
        valid = ~np.isnan(fut) & ~np.isnan(rc)
        if valid.sum() == 0:
            continue

        actual_log_ret = np.log((fut[valid] / rc[valid]).clip(1e-9))
        pred_log_ret   = preds_return[valid, h_idx]

        pred_up   = pred_log_ret > 0
        actual_up = actual_log_ret > 0
        n_correct = (pred_up == actual_up).sum()
        dir_acc   = 100.0 * n_correct / valid.sum()
        rmse      = np.sqrt(np.nanmean((pred_log_ret - actual_log_ret) ** 2))
        mae       = np.nanmean(np.abs(pred_log_ret - actual_log_ret))

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
        arrow = "UP  ^" if preds_return[-1, h_idx] > 0 else "DOWN v"
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

    run_test(args.symbol, args.start, args.end, show_all=args.all)
