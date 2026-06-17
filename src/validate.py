import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score

CACHE_DIR  = "data/cache"
MODEL_PATH = "models/best_lstm_attention.pt"
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


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_horizons):
        super().__init__()
        lstm_drop = dropout if num_layers > 1 else 0.0
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers,
                                 batch_first=True, dropout=lstm_drop)
        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.cls_head  = nn.Linear(hidden_size, num_horizons)

    def forward(self, x):
        lstm_out, _      = self.lstm(x)
        context, weights = self.attention(lstm_out)
        z                = self.dropout(context)
        return self.cls_head(z), weights


def run_validation():
    print(f"Loading model from {MODEL_PATH} ...")
    ckpt        = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    input_size  = ckpt["feature_cols_count"]
    hidden_size = ckpt.get("hidden_size", 64)
    num_layers  = ckpt.get("num_layers",  1)
    dropout     = ckpt.get("dropout",     0.3)
    HORIZONS    = ckpt.get("horizons",    [20, 30, 40])

    model = LSTMClassifier(input_size, hidden_size, num_layers, dropout, len(HORIZONS)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Device: {DEVICE}  |  Horizons: {HORIZONS}")

    target_scalers = ckpt.get("target_scalers", {})

    print(f"\nLoading test data from {CACHE_DIR} ...")
    test_X  = np.load(f"{CACHE_DIR}/test_X.npy")
    test_y  = np.load(f"{CACHE_DIR}/test_y.npy")
    print(f"  Test sequences: {len(test_X):,}")

    all_cls = []
    with torch.no_grad():
        for i in range(0, len(test_X), BATCH_SIZE):
            X_batch = torch.tensor(test_X[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            cls_logits, _ = model(X_batch)
            all_cls.append(cls_logits.cpu().numpy())

    cls_arr  = np.concatenate(all_cls)
    tgts_arr = test_y

    def invert(arr_std, h_idx):
        h = HORIZONS[h_idx]
        if h in target_scalers:
            return target_scalers[h].inverse_transform(arr_std.reshape(-1, 1)).ravel()
        return arr_std

    # ── Direction Classifier ───────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  DIRECTION CLASSIFIER — TEST RESULTS")
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

    # ── Collapse check ─────────────────────────────────────────────────────
    print(f"\n  COLLAPSE CHECK  (clsUP% should be 40-60%)")
    print(f"  {'Horizon':>10}  {'clsUP%':>8}  {'P(UP) mean':>12}  {'P(UP) std':>11}  {'status':>10}")
    print(f"  {'─' * 56}")
    for i, h in enumerate(HORIZONS):
        up_pct = (cls_arr[:, i] > 0).mean() * 100
        p_up   = 1.0 / (1.0 + np.exp(-cls_arr[:, i]))
        status = "OK" if 35 < up_pct < 65 else "COLLAPSED"
        print(f"  {h:>8d}d  {up_pct:>7.1f}%  {p_up.mean():>12.4f}  {p_up.std():>11.4f}  {status:>10}")

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
            logits, _ = model(Xb)
            val_cls_raw.append(logits.cpu().numpy())
    val_cls_arr = np.concatenate(val_cls_raw)

    THRESHOLDS = np.round(np.arange(0.25, 0.76, 0.05), 2)
    best_thrs  = {}

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

        test_p    = 1.0 / (1.0 + np.exp(-cls_arr[:, i]))
        test_true = (invert(tgts_arr[:, i], i) > 0).astype(int)
        base_test = test_true.mean() * 100
        skill_50  = ((test_p >= 0.50).astype(int) == test_true).mean() * 100 - base_test
        skill_bt  = ((test_p >= best_thr).astype(int) == test_true).mean() * 100 - base_test
        print(f"\n  Best val thr={best_thr:.2f}  →  Test Skill @ 0.50: {skill_50:+.2f}%"
              f"   @ {best_thr:.2f}: {skill_bt:+.2f}%")

    # ── Classification summary at best threshold ───────────────────────────
    print(f"\n{'=' * 76}")
    print("  CLASSIFICATION SUMMARY  (best threshold per horizon)")
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
