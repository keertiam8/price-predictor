"""
validate_timesfm.py — validate TimesFM on the cached test split.

Mirrors validate.py output format exactly.
Run train_timesfm.py first to build the cache.

Usage:
    python src/validate_timesfm.py
"""
import os
import sys
import pickle
import numpy as np
import torch

import timesfm
try:
    from importlib.metadata import version as pkg_version
    TIMESFM_VERSION = pkg_version('timesfm')
except Exception:
    TIMESFM_VERSION = 'unknown'

CACHE_DIR    = "data/cache_timesfm"
HORIZONS     = [5, 10, 20]
HORIZON_LEN  = max(HORIZONS)
BATCH_SIZE   = 128
DEVICE_STR   = "gpu" if torch.cuda.is_available() else "cpu"
TIMESFM_REPO = "google/timesfm-2.0-500m-pytorch"


def run_validation():
    print(f"Loading TimesFM from {TIMESFM_REPO}  (device: {DEVICE_STR}) ...")
    config = timesfm.ForecastConfig(
        horizon_len=HORIZON_LEN,
        backend=DEVICE_STR,
        per_core_batch_size=BATCH_SIZE,
    )
    tfm = timesfm.TimesFM_2p5_200M_torch(config=config)
    tfm.load_from_checkpoint(repo_id=TIMESFM_REPO)
    print("  TimesFM loaded.")

    print(f"\nLoading test data from {CACHE_DIR} ...")
    test_lr = np.load(f"{CACHE_DIR}/test_logret.npy")
    test_y  = np.load(f"{CACHE_DIR}/test_y.npy")
    print(f"  Test sequences: {len(test_lr):,}")

    # Batched inference
    all_forecasts = []
    for start in range(0, len(test_lr), BATCH_SIZE):
        batch = test_lr[start : start + BATCH_SIZE]
        inputs = [batch[i] for i in range(len(batch))]
        point_forecast, _ = tfm.forecast(inputs, freq=[0] * len(inputs))
        all_forecasts.append(point_forecast)

    forecasts = np.concatenate(all_forecasts, axis=0)   # (N, HORIZON_LEN)

    # Cumulative sum → multi-step log-return predictions
    preds = np.stack([forecasts[:, :h].sum(axis=1) for h in HORIZONS], axis=1)

    def horizon_block(h_idx, h):
        p = preds[:, h_idx]
        t = test_y[:, h_idx]
        mask = ~np.isnan(t) & ~np.isnan(p)
        p, t = p[mask], t[mask]

        rmse    = np.sqrt(np.mean((p - t) ** 2))
        mae     = np.mean(np.abs(p - t))
        dir_acc = np.mean((p > 0) == (t > 0)) * 100

        p_pct = (np.exp(p) - 1) * 100
        t_pct = (np.exp(t) - 1) * 100

        print(f"\n{'='*60}")
        print(f"  {h}-DAY HORIZON")
        print(f"{'='*60}")
        print(f"  RMSE (log-return)    : {rmse:>10.4f}")
        print(f"  MAE  (log-return)    : {mae:>10.4f}")
        print(f"  Directional Acc      : {dir_acc:>9.2f}%")
        print(f"  (Dir: predicted return > 0 == actual return > 0)")
        print(f"{'─'*60}")
        print(f"  Sample predictions (first 8 test sequences):")
        print(f"  {'#':>4}  {'Actual %':>10}  {'Pred %':>10}  {'Err %':>9}  {'Correct':>7}")
        print(f"  {'─'*52}")
        for j in range(min(8, len(p))):
            err     = p_pct[j] - t_pct[j]
            correct = (p[j] > 0) == (t[j] > 0)
            tick    = "YES" if correct else "NO"
            print(f"  {j+1:>4}  {t_pct[j]:>+9.2f}%  {p_pct[j]:>+9.2f}%  {err:>+8.2f}%  {tick:>7}")

        return rmse, mae, dir_acc

    all_rmse, all_mae, all_dir = [], [], []
    for i, h in enumerate(HORIZONS):
        r, m, d = horizon_block(i, h)
        all_rmse.append(r); all_mae.append(m); all_dir.append(d)

    print(f"\n{'='*52}")
    print(f"  COMBINED SUMMARY (all horizons)")
    print(f"{'='*52}")
    print(f"  {'Horizon':>10}  {'RMSE':>8}  {'MAE':>8}  {'Dir Acc':>9}")
    print(f"  {'─'*46}")
    for i, h in enumerate(HORIZONS):
        print(f"  {h:>8d}d  {all_rmse[i]:>8.4f}  {all_mae[i]:>8.4f}  {all_dir[i]:>8.2f}%")
    print(f"  {'─'*46}")
    print(f"  {'Average':>10}  {np.mean(all_rmse):>8.4f}  {np.mean(all_mae):>8.4f}"
          f"  {np.mean(all_dir):>8.2f}%")
    print(f"{'='*52}")
    print(f"\n  RMSE/MAE are in log-return units. Dir Acc = fraction of"
          f"\n  sequences where predicted return direction matches actual.")


if __name__ == "__main__":
    if not os.path.exists(f"{CACHE_DIR}/test_logret.npy"):
        print(f"No cache at {CACHE_DIR} — run train_timesfm.py first.")
        sys.exit(1)
    run_validation()
