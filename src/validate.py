import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

CACHE_DIR  = "data/cache"
MODEL_PATH = "models/best_lstm_attention.pt"
HORIZONS   = [5, 10, 20]
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


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"No trained model at {MODEL_PATH} — run train.py first.")
    elif not os.path.exists(f"{CACHE_DIR}/test_X.npy"):
        print(f"No cache at {CACHE_DIR} — run train.py first.")
    else:
        run_validation()
