"""
train_timesfm.py — zero-shot evaluation of Google TimesFM on the same
train/val/test splits used by train.py.

No training occurs; TimesFM is used as a pre-trained foundation model.
Preprocessing mirrors train.py exactly (same parquet, same splits, same targets).
TimesFM receives the last LOOKBACK daily log-returns as context and forecasts
the next max(HORIZONS)=20 steps. Multi-horizon targets are derived as cumulative
sums: 5d = sum(forecasts[:5]), 10d = sum(forecasts[:10]), 20d = sum(forecasts[:20]).

Results and preprocessing metadata are cached to data/cache_timesfm/ for use
by validate_timesfm.py and test_timesfm.py.

Usage:
    python src/train_timesfm.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import timesfm

# ── Config (mirrors train.py) ─────────────────────────────────────────────
DATA_PATH   = "data/combined_features.parquet"
CACHE_DIR   = "data/cache_timesfm"
LOOKBACK    = 60
HORIZONS    = [5, 10, 20]
HORIZON_LEN = max(HORIZONS)   # TimesFM forecasts this many steps ahead
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
BATCH_SIZE  = 128
DROP_COLS   = ["date", "company_name", "industry"]

TIMESFM_REPO = "google/timesfm-1.0-200m-pytorch"
DEVICE_STR   = "gpu" if torch.cuda.is_available() else "cpu"

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


def load_and_preprocess():
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = df.dropna(subset=["close"])

    for col in ["symbol", "sector", "cap_category"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    raw_close = df[["symbol", "date", "close"]].copy()
    df = _transform_to_returns(df)
    df = df.merge(raw_close.rename(columns={"close": "_raw_close"}),
                  on=["symbol", "date"], how="left")

    dates     = np.sort(df["date"].unique())
    train_cut = dates[int(len(dates) * TRAIN_RATIO)]
    val_cut   = dates[int(len(dates) * (TRAIN_RATIO + VAL_RATIO))]

    train_df = df[df["date"] <  train_cut].copy()
    val_df   = df[(df["date"] >= train_cut) & (df["date"] < val_cut)].copy()
    test_df  = df[df["date"] >= val_cut].copy()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    train_df[numeric_cols] = train_df.groupby("symbol")[numeric_cols].transform(
        lambda x: x.ffill().bfill()
    )
    val_df[numeric_cols]  = val_df.groupby("symbol")[numeric_cols].transform(lambda x: x.ffill())
    test_df[numeric_cols] = test_df.groupby("symbol")[numeric_cols].transform(lambda x: x.ffill())

    # Compute raw log-return targets (no StandardScaler — TimesFM works in original space)
    for split in [train_df, val_df, test_df]:
        for h in HORIZONS:
            future_close = split.groupby("symbol")["_raw_close"].shift(-h)
            cur_close    = split["_raw_close"].replace(0, np.nan)
            split[f"target_{h}d"] = np.log((future_close / cur_close).clip(1e-9))

    test_start = test_df["date"].min()
    test_end   = test_df["date"].max()

    print(f"  Train: {len(train_df):,}  ({train_df['date'].min().date()} -> {train_df['date'].max().date()})")
    print(f"  Val  : {len(val_df):,}  ({val_df['date'].min().date()} -> {val_df['date'].max().date()})")
    print(f"  Test : {len(test_df):,}  ({test_df['date'].min().date()} -> {test_df['date'].max().date()})")

    return train_df, val_df, test_df, test_start, test_end


def build_sequences(split_df):
    """Build (log_return context, targets, dates, sym_ids) arrays."""
    target_cols = [f"target_{h}d" for h in HORIZONS]
    split_df = split_df.dropna(subset=target_cols).copy()

    logret_seqs, targets, dates, sym_ids = [], [], [], []

    for sym_id, grp in split_df.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)
        lr  = grp["log_return"].values.astype(np.float32)
        tgt = grp[target_cols].values.astype(np.float32)

        for i in range(LOOKBACK, len(grp)):
            logret_seqs.append(lr[i - LOOKBACK : i])
            targets.append(tgt[i])
            dates.append(grp["date"].iloc[i])
            sym_ids.append(sym_id)

    return (np.array(logret_seqs, dtype=np.float32),
            np.array(targets,     dtype=np.float32),
            dates,
            np.array(sym_ids,     dtype=np.int64))


def run_timesfm_inference(tfm, logret_seqs):
    """Run TimesFM on batches of log-return sequences.

    Returns preds shape (N, len(HORIZONS)) where each column is the
    cumulative sum of forecasted daily log-returns up to that horizon.
    """
    n = len(logret_seqs)
    all_forecasts = []

    for start in range(0, n, BATCH_SIZE):
        batch = logret_seqs[start : start + BATCH_SIZE]
        inputs = [batch[i] for i in range(len(batch))]
        point_forecast, _ = tfm.forecast(inputs, freq=[0] * len(inputs))
        all_forecasts.append(point_forecast)   # (batch, HORIZON_LEN)

    forecasts = np.concatenate(all_forecasts, axis=0)   # (N, HORIZON_LEN)

    # Cumulative sum over horizon steps → multi-step log-return predictions
    preds = np.stack([
        forecasts[:, :h].sum(axis=1) for h in HORIZONS
    ], axis=1)   # (N, 3)
    return preds


def print_metrics(split_name, preds, targets):
    """Print metrics in same format as train.py final test block."""
    print(f"\n{'='*60}")
    print(f"  {split_name} set metrics (raw log-return space):")
    print(f"  {'Horizon':>10}  {'RMSE(ret)':>10}  {'MAE(ret)':>9}  {'Dir Acc':>9}")
    print(f"  {'─'*44}")
    all_rmse, all_mae, all_dir = [], [], []
    for i, h in enumerate(HORIZONS):
        p = preds[:, i]
        t = targets[:, i]
        mask = ~np.isnan(t) & ~np.isnan(p)
        p, t = p[mask], t[mask]

        rmse    = np.sqrt(np.mean((p - t) ** 2))
        mae     = np.mean(np.abs(p - t))
        dir_acc = np.mean((p > 0) == (t > 0)) * 100

        all_rmse.append(rmse); all_mae.append(mae); all_dir.append(dir_acc)
        print(f"  {h:>8d}d  {rmse:>10.4f}  {mae:>9.4f}  {dir_acc:>8.2f}%")

    print(f"  {'─'*44}")
    print(f"  {'Average':>10}  {np.mean(all_rmse):>10.4f}  {np.mean(all_mae):>9.4f}"
          f"  {np.mean(all_dir):>8.2f}%")
    print(f"{'='*60}")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Loading and preprocessing data...")
    train_df, val_df, test_df, test_start, test_end = load_and_preprocess()

    print("\nBuilding sequences...")
    train_lr, train_y, _, _         = build_sequences(train_df)
    val_lr,   val_y,   _, _         = build_sequences(val_df)
    test_lr,  test_y,  test_dates, test_sym_ids = build_sequences(test_df)
    print(f"  Train: {len(train_lr):,} | Val: {len(val_lr):,} | Test: {len(test_lr):,} sequences")

    print(f"\nLoading TimesFM from {TIMESFM_REPO}  (device: {DEVICE_STR}) ...")
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=DEVICE_STR,
            per_core_batch_size=BATCH_SIZE,
            horizon_len=HORIZON_LEN,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id=TIMESFM_REPO,
        ),
    )
    print("  TimesFM loaded.")

    print("\nRunning inference on train split...")
    train_preds = run_timesfm_inference(tfm, train_lr)
    print_metrics("TRAIN", train_preds, train_y)

    print("\nRunning inference on val split...")
    val_preds = run_timesfm_inference(tfm, val_lr)
    print_metrics("VAL", val_preds, val_y)

    print("\nRunning inference on test split...")
    test_preds = run_timesfm_inference(tfm, test_lr)
    print_metrics("TEST", test_preds, test_y)

    # ── Save cache for validate_timesfm.py and test_timesfm.py ───────────
    np.save(f"{CACHE_DIR}/test_logret.npy",   test_lr)
    np.save(f"{CACHE_DIR}/test_y.npy",        test_y)
    np.save(f"{CACHE_DIR}/test_sym_ids.npy",  test_sym_ids)
    with open(f"{CACHE_DIR}/meta.pkl", "wb") as f:
        pickle.dump({
            "test_start_date": str(test_start.date()),
            "test_end_date":   str(test_end.date()),
            "test_dates":      [str(d.date()) for d in test_dates],
        }, f)
    print(f"\nCache saved to {CACHE_DIR}/")


if __name__ == "__main__":
    main()
