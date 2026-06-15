"""
Experiment A — Classification-only LSTM+Attention.

Identical to train.py except:
  - No reg_head, no reg_loss, no Huber magnitude loss
  - Loss = pure BCEWithLogitsLoss on direction
  - Saved to models/best_cls_only.pt (does NOT overwrite train.py's model)

Run after train.py has built the cache:
    python src/train_cls.py

If cls-only Val Skill >> two-stage Val Skill, the reg_head was interfering.
If they match, the issue is features / regime, not architecture.
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

# ── Config ──────────────────────────────────────────────────────────────────
CACHE_DIR  = "data/cache"
MODEL_DIR  = "models"
MODEL_PATH = "models/best_cls_only.pt"
HORIZONS   = [3, 7, 14]
BATCH_SIZE = 64
EPOCHS     = 100
LR         = 1e-3
HIDDEN_SIZE = 256
NUM_LAYERS  = 3
DROPOUT     = 0.2
EARLY_STOP  = 40
WEIGHT_DECAY = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ─────────────────────────────────────────────────────────────────
class CachedDataset(Dataset):
    def __init__(self, X, y, cc):
        self.X  = torch.tensor(X,  dtype=torch.float32)
        self.y  = torch.tensor(y,  dtype=torch.float32)
        self.cc = torch.tensor(cc, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.cc[idx]


# ── Model ────────────────────────────────────────────────────────────────────
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        e = self.score(lstm_out).squeeze(-1)
        a = torch.softmax(e, dim=1).unsqueeze(-1)
        return (a * lstm_out).sum(dim=1), a.squeeze(-1)


class ClsOnlyModel(nn.Module):
    """LSTM + Attention → direction logits only. No magnitude head."""
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_horizons):
        super().__init__()
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0.0)
        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.cls_head  = nn.Linear(hidden_size, num_horizons)

    def forward(self, x):
        lstm_out, _      = self.lstm(x)
        context, weights = self.attention(lstm_out)
        z          = self.dropout(context)
        cls_logits = self.cls_head(z)
        return cls_logits, weights


# ── Training helpers ─────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, zero_thresh, pos_weight, training):
    model.train() if training else model.eval()
    total_loss = 0.0
    zt = zero_thresh.unsqueeze(0)
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, y, _ in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            cls_logits, _ = model(X)
            y_dir = (y > zt).float()
            loss  = F.binary_cross_entropy_with_logits(
                        cls_logits, y_dir, pos_weight=pos_weight)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


def directional_accuracy(model, loader, zero_thresh):
    model.eval()
    correct  = torch.zeros(len(HORIZONS))
    baseline = torch.zeros(len(HORIZONS))
    total    = 0
    zt       = zero_thresh.cpu().unsqueeze(0)
    with torch.no_grad():
        for X, y, _ in loader:
            logits, _ = model(X.to(DEVICE))
            logits    = logits.cpu()
            y_up      = (y > zt)
            correct  += ((logits > 0) == y_up).float().sum(dim=0)
            baseline += y_up.float().sum(dim=0)
            total    += len(y)
    per_acc      = (correct  / total * 100).tolist()
    per_baseline = (baseline / total * 100).tolist()
    per_skill    = [a - b for a, b in zip(per_acc, per_baseline)]
    return float(np.mean(per_skill)), float(np.mean(per_acc)), per_acc, per_baseline


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    files = ["train_X.npy", "train_y.npy", "train_cc.npy",
             "val_X.npy",   "val_y.npy",   "val_cc.npy",
             "test_X.npy",  "test_y.npy",  "test_cc.npy", "meta.pkl"]
    if not all(os.path.exists(f"{CACHE_DIR}/{f}") for f in files):
        print(f"No cache at {CACHE_DIR} — run train.py first to build it.")
        return

    print(f"Loading cache from {CACHE_DIR} ...")
    train_X  = np.load(f"{CACHE_DIR}/train_X.npy")
    train_y  = np.load(f"{CACHE_DIR}/train_y.npy")
    train_cc = np.load(f"{CACHE_DIR}/train_cc.npy")
    val_X    = np.load(f"{CACHE_DIR}/val_X.npy")
    val_y    = np.load(f"{CACHE_DIR}/val_y.npy")
    val_cc   = np.load(f"{CACHE_DIR}/val_cc.npy")
    test_X   = np.load(f"{CACHE_DIR}/test_X.npy")
    test_y   = np.load(f"{CACHE_DIR}/test_y.npy")
    test_cc  = np.load(f"{CACHE_DIR}/test_cc.npy")
    with open(f"{CACHE_DIR}/meta.pkl", "rb") as f:
        meta = pickle.load(f)
    target_scalers = meta["target_scalers"]

    train_ds = CachedDataset(train_X, train_y, train_cc)
    val_ds   = CachedDataset(val_X,   val_y,   val_cc)
    test_ds  = CachedDataset(test_X,  test_y,  test_cc)

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"  Window: {train_X.shape[1]} steps × {train_X.shape[2]} features")

    zero_thresh = torch.tensor(
        [float(target_scalers[h].transform([[0.0]])[0, 0]) for h in HORIZONS],
        dtype=torch.float32, device=DEVICE,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    feature_cols_count = train_X.shape[2]
    model = ClsOnlyModel(
        input_size   = feature_cols_count,
        hidden_size  = HIDDEN_SIZE,
        num_layers   = NUM_LAYERS,
        dropout      = DROPOUT,
        num_horizons = len(HORIZONS),
    ).to(DEVICE)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters | Device: {DEVICE}")

    # pos_weight — same formula as train.py
    zt_cpu    = zero_thresh.cpu()
    train_y_t = train_ds.y
    n_up   = (train_y_t > zt_cpu.unsqueeze(0)).float().sum(dim=0).clamp(min=1)
    n_down = (train_y_t <= zt_cpu.unsqueeze(0)).float().sum(dim=0).clamp(min=1)
    pos_weight = ((n_down / n_up) * 0.95).to(DEVICE)
    print(f"  pos_weight (0.95 × n_down/n_up): "
          + " / ".join(f"{h}d={v:.3f}" for h, v in zip(HORIZONS, pos_weight.tolist())))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val_skill = -999.0
    no_improve     = 0

    print(f"\nTraining — early stop patience={EARLY_STOP}  |  checkpoint on val SKILL")
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>9}  {'ValAcc':>8}"
          f"  {'Baseline':>9}  {'Skill':>7}  {'/'.join(str(h)+'d' for h in HORIZONS):>16}")
    print("-" * 88)

    for epoch in range(1, EPOCHS + 1):
        tr_loss = run_epoch(model, train_loader, optimizer, zero_thresh, pos_weight, training=True)
        va_loss = run_epoch(model, val_loader,   optimizer, zero_thresh, pos_weight, training=False)
        skill, acc_avg, acc_per, base_per = directional_accuracy(model, val_loader, zero_thresh)
        scheduler.step(va_loss)

        marker = ""
        if skill > best_val_skill:
            best_val_skill = skill
            no_improve     = 0
            torch.save({"model_state":        model.state_dict(),
                        "target_scalers":     target_scalers,
                        "feature_cols":       meta["feature_cols"],
                        "feature_cols_count": feature_cols_count,
                        "hidden_size":        HIDDEN_SIZE,
                        "num_layers":         NUM_LAYERS,
                        "dropout":            DROPOUT}, MODEL_PATH)
            marker = " *"
        else:
            no_improve += 1

        base_avg = float(np.mean(base_per))
        per_str  = "/".join(f"{a:.1f}" for a in acc_per)
        print(f"{epoch:>6d}  {tr_loss:>10.6f}  {va_loss:>9.6f}  {acc_avg:>7.2f}%"
              f"  {base_avg:>8.2f}%  {skill:>+6.2f}%  {per_str:>16}{marker}")

        if no_improve >= EARLY_STOP:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest Val Skill: {best_val_skill:+.2f}%  → saved to {MODEL_PATH}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_cls, all_tgts = [], []
    with torch.no_grad():
        for X, y, _ in test_loader:
            logits, _ = model(X.to(DEVICE))
            all_cls.append(logits.cpu().numpy())
            all_tgts.append(y.numpy())

    cls_arr  = np.concatenate(all_cls)
    tgts_arr = np.concatenate(all_tgts)
    zt_np    = np.array([float(target_scalers[h].transform([[0.0]])[0, 0]) for h in HORIZONS])

    def invert(arr_std, h_idx):
        h = HORIZONS[h_idx]
        return target_scalers[h].inverse_transform(arr_std.reshape(-1, 1)).ravel()

    # ── Direction results ─────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  EXPERIMENT A — CLS-ONLY  (test set)")
    print("=" * 74)
    print(f"  {'Horizon':>10}  {'Dir Acc':>9}  {'Baseline':>9}  {'Skill':>7}"
          f"  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("  " + "-" * 68)
    for i, h in enumerate(HORIZONS):
        t_raw  = invert(tgts_arr[:, i], i)
        y_true = (t_raw > 0).astype(int)
        y_pred = (cls_arr[:, i] > 0).astype(int)
        acc    = np.mean(y_true == y_pred) * 100
        base   = y_true.mean() * 100
        skill  = acc - base
        prec   = precision_score(y_true, y_pred, zero_division=0) * 100
        rec    = recall_score(y_true, y_pred, zero_division=0) * 100
        f1     = f1_score(y_true, y_pred, zero_division=0) * 100
        print(f"  {h:>8d}d  {acc:>8.2f}%  {base:>8.2f}%  {skill:>+6.2f}%"
              f"  {prec:>9.2f}%  {rec:>7.2f}%  {f1:>7.2f}%")

    # ── Probability distribution ──────────────────────────────────────────────
    print(f"\n{'=' * 58}")
    print("  PROBABILITY DISTRIBUTION  P(UP) = sigmoid(cls_logit)")
    print(f"{'=' * 58}")
    for i, h in enumerate(HORIZONS):
        p    = 1.0 / (1.0 + np.exp(-cls_arr[:, i]))
        pcts = np.percentile(p, [10, 25, 50, 75, 90])
        print(f"\n  {h}d  mean={p.mean():.3f}  std={p.std():.3f}"
              f"  p10={pcts[0]:.2f}  p50={pcts[2]:.2f}  p90={pcts[4]:.2f}")
        counts, edges = np.histogram(p, bins=10, range=(0.0, 1.0))
        mx = max(counts) or 1
        for lo, hi, cnt in zip(edges[:-1], edges[1:], counts):
            bar = "█" * round(cnt / mx * 28)
            print(f"    {lo:.1f}–{hi:.1f} |{bar:<28}| {cnt:5d}")

    # ── clsUP% check ─────────────────────────────────────────────────────────
    print(f"\n  {'Horizon':>10}  {'clsUP%':>8}  {'std(P)':>8}  {'status':>10}")
    print(f"  {'─' * 44}")
    for i, h in enumerate(HORIZONS):
        p      = 1.0 / (1.0 + np.exp(-cls_arr[:, i]))
        up_pct = (cls_arr[:, i] > 0).mean() * 100
        status = "biased" if up_pct < 35 or up_pct > 65 else "OK"
        print(f"  {h:>8d}d  {up_pct:>7.1f}%  {p.std():>8.4f}  {status:>10}")

    print(f"\n{'=' * 74}")
    print("  Compare Best Val Skill here vs train.py to see if reg_head was hurting.")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
