import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

CACHE_DIR  = "data/cache"
MODEL_PATH = "models/best_lstm_attention.pt"
HORIZONS   = [3, 7, 14]
BATCH_SIZE = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        lstm_drop = dropout if num_layers > 1 else 0.0
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers,
                                 batch_first=True, dropout=lstm_drop)
        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.cls_head  = nn.Linear(hidden_size, num_horizons)
        self.reg_head  = nn.Linear(hidden_size + num_horizons, num_horizons)

    def forward(self, x):
        lstm_out, _      = self.lstm(x)
        context, weights = self.attention(lstm_out)
        z                = self.dropout(context)
        cls_logits       = self.cls_head(z)
        cls_probs        = torch.sigmoid(cls_logits).detach()
        mag_preds        = F.relu(self.reg_head(torch.cat([z, cls_probs], dim=-1)))
        return cls_logits, mag_preds, weights


def run_validation():
    print(f"Loading model from {MODEL_PATH} ...")
    ckpt        = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    input_size  = ckpt["feature_cols_count"]
    hidden_size = ckpt.get("hidden_size", 256)
    num_layers  = ckpt.get("num_layers",  3)
    dropout     = ckpt.get("dropout",     0.2)

    model = TwoStageModel(input_size, hidden_size, num_layers, dropout, len(HORIZONS)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Device: {DEVICE}")

    target_scalers = ckpt.get("target_scalers", {})
    zt_np = np.array([float(target_scalers[h].transform([[0.0]])[0, 0]) for h in HORIZONS])

    print(f"\nLoading test data from {CACHE_DIR} ...")
    test_X  = np.load(f"{CACHE_DIR}/test_X.npy")
    test_y  = np.load(f"{CACHE_DIR}/test_y.npy")
    print(f"  Test sequences: {len(test_X):,}")

    all_cls, all_mag = [], []
    with torch.no_grad():
        for i in range(0, len(test_X), BATCH_SIZE):
            X_batch = torch.tensor(test_X[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            cls_logits, mag_preds, _ = model(X_batch)
            all_cls.append(cls_logits.cpu().numpy())
            all_mag.append(mag_preds.cpu().numpy())

    cls_arr  = np.concatenate(all_cls)   # (N, H)
    mag_arr  = np.concatenate(all_mag)   # (N, H)
    tgts_arr = test_y                    # (N, H) standardised

    def invert(arr_std, h_idx):
        h = HORIZONS[h_idx]
        if h in target_scalers:
            return target_scalers[h].inverse_transform(arr_std.reshape(-1, 1)).ravel()
        return arr_std

    # ── Stage 1: Direction Classifier ──────────────────────────────────────
    print("\n" + "=" * 76)
    print("  STAGE 1 — DIRECTION CLASSIFIER")
    print("=" * 76)
    print(f"  {'Horizon':>10}  {'Dir Acc':>9}  {'Baseline':>9}  {'Skill':>7}"
          f"  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("  " + "-" * 68)
    for i, h in enumerate(HORIZONS):
        t_raw    = invert(tgts_arr[:, i], i)
        y_true   = (t_raw > 0).astype(int)
        y_pred   = (cls_arr[:, i] > 0).astype(int)
        acc      = np.mean(y_true == y_pred) * 100
        baseline = y_true.mean() * 100
        skill    = acc - baseline
        prec     = precision_score(y_true, y_pred, zero_division=0) * 100
        rec      = recall_score(y_true, y_pred, zero_division=0) * 100
        f1       = f1_score(y_true, y_pred, zero_division=0) * 100
        print(f"  {h:>8d}d  {acc:>8.2f}%  {baseline:>8.2f}%  {skill:>+6.2f}%"
              f"  {prec:>9.2f}%  {rec:>7.2f}%  {f1:>7.2f}%")

    # ── Stage 2: Magnitude Regressor ───────────────────────────────────────
    print("\n" + "=" * 66)
    print("  STAGE 2 — MAGNITUDE REGRESSOR  (std space)")
    print("=" * 66)
    print(f"  {'Horizon':>10}  {'RMSE(mag)':>10}  {'MAE(mag)':>10}  {'MagMean':>9}  {'MagStd':>8}")
    print("  " + "-" * 52)
    for i, h in enumerate(HORIZONS):
        y_mag_true = np.abs(tgts_arr[:, i] - zt_np[i])
        y_mag_pred = mag_arr[:, i]
        rmse = np.sqrt(np.nanmean((y_mag_pred - y_mag_true) ** 2))
        mae  = np.nanmean(np.abs(y_mag_pred - y_mag_true))
        print(f"  {h:>8d}d  {rmse:>10.4f}  {mae:>10.4f}  {y_mag_pred.mean():>+9.4f}"
              f"  {y_mag_pred.std():>8.4f}")

    # ── Combined prediction ─────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  COMBINED PREDICTION  sign(cls) × magnitude")
    print("=" * 76)
    print(f"  {'Horizon':>10}  {'RMSE(ret)':>10}  {'MAE(ret)':>9}"
          f"  {'Dir Acc':>9}  {'Baseline':>9}  {'Skill':>7}")
    print("  " + "-" * 62)
    all_rmse, all_mae, all_dir = [], [], []
    for i, h in enumerate(HORIZONS):
        sign      = np.where(cls_arr[:, i] > 0, 1.0, -1.0)
        final_std = zt_np[i] + sign * mag_arr[:, i]
        pred_raw  = invert(final_std, i)
        true_raw  = invert(tgts_arr[:, i], i)
        rmse      = np.sqrt(np.nanmean((pred_raw - true_raw) ** 2))
        mae       = np.nanmean(np.abs(pred_raw - true_raw))
        dir_acc   = np.mean((pred_raw > 0) == (true_raw > 0)) * 100
        baseline  = (true_raw > 0).mean() * 100
        skill     = dir_acc - baseline

        pred_pct = (np.exp(pred_raw) - 1) * 100
        true_pct = (np.exp(true_raw) - 1) * 100

        print(f"  {h:>8d}d  {rmse:>10.4f}  {mae:>9.4f}"
              f"  {dir_acc:>8.2f}%  {baseline:>8.2f}%  {skill:>+6.2f}%")
        all_rmse.append(rmse); all_mae.append(mae); all_dir.append(dir_acc)

        # Sample rows
        print(f"  {'─' * 56}")
        print(f"  Sample predictions (first 8):")
        print(f"  {'#':>4}  {'Actual %':>10}  {'Pred %':>10}  {'Dir':>4}  {'Correct':>7}")
        print(f"  {'─' * 44}")
        pred_up = cls_arr[:, i] > 0
        for j in range(min(8, len(pred_raw))):
            d_str   = "UP" if pred_up[j] else "DN"
            correct = "YES" if (pred_raw[j] > 0) == (true_raw[j] > 0) else "NO"
            print(f"  {j+1:>4}  {true_pct[j]:>+9.2f}%  {pred_pct[j]:>+9.2f}%  {d_str:>4}  {correct:>7}")
        print()

    print("=" * 76)
    print(f"  {'AVERAGE':>10}  {np.mean(all_rmse):>10.4f}  {np.mean(all_mae):>9.4f}"
          f"  {np.mean(all_dir):>8.2f}%")
    print("=" * 76)

    # ── Collapse check ─────────────────────────────────────────────────────
    print(f"\n  COLLAPSE CHECK  (magnitude std should be >0.3; %UP should be 40-60%)")
    print(f"  {'Horizon':>10}  {'MagMean':>9}  {'MagStd':>8}  {'clsUP%':>8}  {'status':>12}")
    print(f"  {'─' * 56}")
    for i, h in enumerate(HORIZONS):
        mag    = mag_arr[:, i]
        mn     = mag.mean()
        std    = mag.std()
        up_pct = (cls_arr[:, i] > 0).mean() * 100
        status = "COLLAPSED" if std < 0.3 else ("biased" if up_pct < 35 or up_pct > 65 else "OK")
        print(f"  {h:>8d}d  {mn:>+9.4f}  {std:>8.4f}  {up_pct:>7.1f}%  {status:>12}")

    # ── Probability distribution ────────────────────────────────────────────
    print(f"\n{'=' * 66}")
    print("  PROBABILITY DISTRIBUTION  P(UP) = sigmoid(cls_logit)")
    print(f"{'=' * 66}")
    for i, h in enumerate(HORIZONS):
        p    = 1.0 / (1.0 + np.exp(-cls_arr[:, i]))
        pcts = np.percentile(p, [10, 25, 50, 75, 90])
        print(f"\n  {h}d  mean={p.mean():.3f}  std={p.std():.3f}"
              f"  p10={pcts[0]:.2f}  p25={pcts[1]:.2f}  p50={pcts[2]:.2f}"
              f"  p75={pcts[3]:.2f}  p90={pcts[4]:.2f}")
        counts, edges = np.histogram(p, bins=10, range=(0.0, 1.0))
        mx = max(counts) or 1
        for lo, hi, cnt in zip(edges[:-1], edges[1:], counts):
            bar = "█" * round(cnt / mx * 28)
            print(f"    {lo:.1f}–{hi:.1f} |{bar:<28}| {cnt:5d}")

    # ── Threshold selection (sweep val, apply to test) ─────────────────────
    print(f"\n{'=' * 66}")
    print("  THRESHOLD SELECTION  (sweep on val → best applied to test)")
    print(f"{'=' * 66}")

    val_X_np    = np.load(f"{CACHE_DIR}/val_X.npy")
    val_y_np    = np.load(f"{CACHE_DIR}/val_y.npy")
    val_cls_raw = []
    with torch.no_grad():
        for j in range(0, len(val_X_np), BATCH_SIZE):
            Xb = torch.tensor(val_X_np[j:j+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            logits, _, _ = model(Xb)
            val_cls_raw.append(logits.cpu().numpy())
    val_cls_arr = np.concatenate(val_cls_raw)

    THRESHOLDS  = np.round(np.arange(0.25, 0.76, 0.05), 2)
    best_thrs   = {}

    for i, h in enumerate(HORIZONS):
        val_p    = 1.0 / (1.0 + np.exp(-val_cls_arr[:, i]))
        val_true = (invert(val_y_np[:, i], i) > 0).astype(int)
        base_val = val_true.mean() * 100

        best_thr, best_sk = 0.50, -999.0
        print(f"\n  {h}d  (val base={base_val:.1f}% UP)")
        print(f"  {'Thresh':>7}  {'Val Acc':>9}  {'Val Skill':>10}  {'%UP pred':>9}")
        print(f"  {'─' * 42}")
        for thr in THRESHOLDS:
            pred = (val_p >= thr).astype(int)
            acc  = (pred == val_true).mean() * 100
            sk   = acc - base_val
            pup  = pred.mean() * 100
            tag  = " ◄" if sk > best_sk else ""
            if sk > best_sk:
                best_sk, best_thr = sk, float(thr)
            print(f"  {thr:>7.2f}  {acc:>8.2f}%  {sk:>+9.2f}%  {pup:>8.1f}%{tag}")
        best_thrs[h] = best_thr

        # Apply best threshold to test
        test_p    = 1.0 / (1.0 + np.exp(-cls_arr[:, i]))
        test_true = (invert(tgts_arr[:, i], i) > 0).astype(int)
        base_test = test_true.mean() * 100
        sk_50     = (test_p >= 0.50).astype(int)
        sk_bt     = (test_p >= best_thr).astype(int)
        skill_50  = (sk_50 == test_true).mean() * 100 - base_test
        skill_bt  = (sk_bt == test_true).mean() * 100 - base_test
        print(f"\n  Best val thr={best_thr:.2f}  →  Test Skill @ 0.50: {skill_50:+.2f}%"
              f"   @ {best_thr:.2f}: {skill_bt:+.2f}%")

    # ── Classification-only summary (best threshold) ───────────────────────
    print(f"\n{'=' * 76}")
    print("  CLASSIFICATION-ONLY SUMMARY  (Stage 1 only, best-threshold per horizon)")
    print(f"{'=' * 76}")
    print(f"  {'Horizon':>10}  {'Threshold':>10}  {'Test Acc':>9}  {'Baseline':>9}"
          f"  {'Skill':>7}  {'Prec':>7}  {'Rec':>7}")
    print(f"  {'─' * 68}")
    for i, h in enumerate(HORIZONS):
        thr       = best_thrs[h]
        test_p    = 1.0 / (1.0 + np.exp(-cls_arr[:, i]))
        test_true = (invert(tgts_arr[:, i], i) > 0).astype(int)
        pred      = (test_p >= thr).astype(int)
        acc       = (pred == test_true).mean() * 100
        base      = test_true.mean() * 100
        skill     = acc - base
        prec      = precision_score(test_true, pred, zero_division=0) * 100
        rec       = recall_score(test_true, pred, zero_division=0) * 100
        print(f"  {h:>8d}d  {thr:>10.2f}  {acc:>8.2f}%  {base:>8.2f}%"
              f"  {skill:>+6.2f}%  {prec:>6.1f}%  {rec:>6.1f}%")
    print(f"{'=' * 76}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"No trained model at {MODEL_PATH} — run train.py first.")
    elif not os.path.exists(f"{CACHE_DIR}/test_X.npy"):
        print(f"No cache at {CACHE_DIR} — run train.py first.")
    else:
        run_validation()
