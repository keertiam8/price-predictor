import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ── Config ───────────────────────────────────────────────────────────────
DATA_PATH    = "data/combined_features.parquet"
MODEL_DIR    = "models"
CACHE_DIR    = "data/cache"
LOOKBACK     = 60
HORIZONS     = [5, 10, 20]
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-3
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.4
EARLY_STOP   = 7
WEIGHT_DECAY = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DROP_COLS = ["date", "company_name", "industry"]

# ── Data ─────────────────────────────────────────────────────────────────
def load_and_preprocess(path, train_ratio=0.70, val_ratio=0.15):
    df = pd.read_parquet(path)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = df.dropna(subset=["close"])

    # Label encode — fit on full data so all symbols/sectors are known
    encoders = {}
    for col in ["symbol", "sector", "cap_category"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  Encoded {col}: {dict(enumerate(le.classes_))}")

    # Split FIRST, then fill — prevents bfill leakage
    dates     = np.sort(df["date"].unique())
    train_cut = dates[int(len(dates) * train_ratio)]
    val_cut   = dates[int(len(dates) * (train_ratio + val_ratio))]

    train_df = df[df["date"] <  train_cut].copy()
    val_df   = df[(df["date"] >= train_cut) & (df["date"] < val_cut)].copy()
    test_df  = df[df["date"] >= val_cut].copy()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Train: ffill + bfill (bfill only fills NaNs at the very start of each symbol's history)
    train_df[numeric_cols] = train_df.groupby("symbol")[numeric_cols].transform(
        lambda x: x.ffill().bfill()
    )
    # Val/Test: ffill only — never look into the future
    val_df[numeric_cols]  = val_df.groupby("symbol")[numeric_cols].transform(lambda x: x.ffill())
    test_df[numeric_cols] = test_df.groupby("symbol")[numeric_cols].transform(lambda x: x.ffill())

    # Drop columns that are entirely NaN in training (can't scale them)
    target_cols  = [f"target_{h}d" for h in HORIZONS]
    feature_cols = [c for c in df.columns if c not in DROP_COLS and not c.startswith("target_")]
    all_nan_cols = [c for c in feature_cols if train_df[c].isna().all()]
    if all_nan_cols:
        print(f"  Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]

    # Add targets after split
    for split in [train_df, val_df, test_df]:
        for h in HORIZONS:
            split[f"target_{h}d"] = split.groupby("symbol")["close"].shift(-h)

    # ── Per-symbol normalization ──────────────────────────────────────────
    # Fit scalers on each symbol's TRAIN data, transform all splits.
    # This fixes cross-symbol price scale differences (BAJFINANCE ₹7000 vs RELIANCE ₹2500).
    symbol_scalers = {}
    for sym_id in sorted(train_df["symbol"].unique()):
        sym_train = train_df[train_df["symbol"] == sym_id][feature_cols].values.astype(np.float32)
        sym_tgt   = train_df[train_df["symbol"] == sym_id].dropna(subset=target_cols)[target_cols].values.astype(np.float32)

        feat_sc = MinMaxScaler(clip=True)
        tgt_sc  = MinMaxScaler(clip=True)
        feat_sc.fit(np.nan_to_num(sym_train, nan=0.0))
        tgt_sc.fit(np.nan_to_num(sym_tgt,   nan=0.0))
        symbol_scalers[sym_id] = {"feat": feat_sc, "tgt": tgt_sc}

    # Apply per-symbol scaling to all splits
    for split_df in [train_df, val_df, test_df]:
        for sym_id, sc in symbol_scalers.items():
            mask = split_df["symbol"] == sym_id
            if mask.sum() == 0:
                continue
            fv = split_df.loc[mask, feature_cols].values.astype(np.float32)
            fv = np.nan_to_num(fv, nan=0.0, posinf=1.0, neginf=0.0)
            split_df.loc[mask, feature_cols] = sc["feat"].transform(fv)

        # Scale targets (only rows that have them)
        for sym_id, sc in symbol_scalers.items():
            mask = split_df["symbol"] == sym_id
            valid = mask & split_df[target_cols].notna().all(axis=1)
            if valid.sum() == 0:
                continue
            tv = split_df.loc[valid, target_cols].values.astype(np.float32)
            tv = np.nan_to_num(tv, nan=0.0)
            split_df.loc[valid, target_cols] = sc["tgt"].transform(tv)

    close_col_idx = feature_cols.index("close")
    return train_df, val_df, test_df, feature_cols, target_cols, symbol_scalers, close_col_idx


# ── Dataset ───────────────────────────────────────────────────────────────
class StockDataset(Dataset):
    """Builds LOOKBACK-length sequences. Data is already scaled at this point."""
    def __init__(self, df, feature_cols, target_cols):
        df = df.dropna(subset=target_cols).copy()

        self.sequences, self.targets, self.current_closes, self.sym_ids = [], [], [], []

        for sym_id, grp in df.groupby("symbol"):
            grp   = grp.reset_index(drop=True)
            feats = grp[feature_cols].values.astype(np.float32)
            tgts  = grp[target_cols].values.astype(np.float32)
            close_idx = feature_cols.index("close")
            for i in range(LOOKBACK, len(grp)):
                self.sequences.append(feats[i - LOOKBACK : i])
                self.targets.append(tgts[i])
                self.current_closes.append(feats[i, close_idx])  # scaled current close
                self.sym_ids.append(sym_id)

        self.sequences      = np.array(self.sequences,      dtype=np.float32)
        self.targets        = np.array(self.targets,        dtype=np.float32)
        self.current_closes = np.array(self.current_closes, dtype=np.float32)
        self.sym_ids        = np.array(self.sym_ids,        dtype=np.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx]),
                torch.tensor(self.targets[idx]),
                torch.tensor(self.current_closes[idx]))


# ── Model ─────────────────────────────────────────────────────────────────
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


# ── Train / Eval ─────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, training):
    model.train() if training else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, y, _ in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred, _ = model(X)
            loss = criterion(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


def directional_accuracy(model, loader):
    """
    Correct formula: actual_dir = future_price > current_price
                     pred_dir   = predicted_price > current_price
    Both compared in scaled space (per-symbol scaling makes this valid).
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y, current_close in loader:
            pred, _ = model(X.to(DEVICE))
            pred    = pred.cpu()
            cur     = current_close.unsqueeze(1)          # (batch, 1)
            pred_dir   = (pred > cur).float()             # predicted goes up?
            actual_dir = (y    > cur).float()             # actual goes up?
            correct += (pred_dir == actual_dir).all(dim=1).sum().item()
            total   += len(y)
    return 100.0 * correct / total if total > 0 else 0.0


# ── Cache ─────────────────────────────────────────────────────────────────
def save_cache(train_ds, val_ds, test_ds, symbol_scalers, feature_cols, close_col_idx,
               test_start=None, test_end=None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(f"{CACHE_DIR}/train_X.npy",  train_ds.sequences)
    np.save(f"{CACHE_DIR}/train_y.npy",  train_ds.targets)
    np.save(f"{CACHE_DIR}/train_cc.npy", train_ds.current_closes)
    np.save(f"{CACHE_DIR}/val_X.npy",    val_ds.sequences)
    np.save(f"{CACHE_DIR}/val_y.npy",    val_ds.targets)
    np.save(f"{CACHE_DIR}/val_cc.npy",   val_ds.current_closes)
    np.save(f"{CACHE_DIR}/test_X.npy",       test_ds.sequences)
    np.save(f"{CACHE_DIR}/test_y.npy",       test_ds.targets)
    np.save(f"{CACHE_DIR}/test_cc.npy",      test_ds.current_closes)
    np.save(f"{CACHE_DIR}/test_sym_ids.npy", test_ds.sym_ids)
    with open(f"{CACHE_DIR}/meta.pkl", "wb") as f:
        pickle.dump({"symbol_scalers":  symbol_scalers,
                     "feature_cols":    feature_cols,
                     "close_col_idx":   close_col_idx,
                     "test_start_date": str(test_start),
                     "test_end_date":   str(test_end)}, f)
    print("  Cache saved to", CACHE_DIR)


def cache_exists():
    files = ["train_X.npy", "train_y.npy", "train_cc.npy",
             "val_X.npy",   "val_y.npy",   "val_cc.npy",
             "test_X.npy",  "test_y.npy",  "test_cc.npy",
             "test_sym_ids.npy", "meta.pkl"]
    return all(os.path.exists(f"{CACHE_DIR}/{f}") for f in files)


def load_cache():
    train_X  = np.load(f"{CACHE_DIR}/train_X.npy")
    train_y  = np.load(f"{CACHE_DIR}/train_y.npy")
    train_cc = np.load(f"{CACHE_DIR}/train_cc.npy")
    val_X    = np.load(f"{CACHE_DIR}/val_X.npy")
    val_y    = np.load(f"{CACHE_DIR}/val_y.npy")
    val_cc   = np.load(f"{CACHE_DIR}/val_cc.npy")
    test_X       = np.load(f"{CACHE_DIR}/test_X.npy")
    test_y       = np.load(f"{CACHE_DIR}/test_y.npy")
    test_cc      = np.load(f"{CACHE_DIR}/test_cc.npy")
    test_sym_ids = np.load(f"{CACHE_DIR}/test_sym_ids.npy")
    with open(f"{CACHE_DIR}/meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return (train_X, train_y, train_cc,
            val_X,   val_y,   val_cc,
            test_X,  test_y,  test_cc, test_sym_ids,
            meta["symbol_scalers"], meta["feature_cols"], meta["close_col_idx"],
            meta.get("test_start_date"), meta.get("test_end_date"))


class CachedDataset(Dataset):
    def __init__(self, X, y, cc):
        self.X  = torch.tensor(X,  dtype=torch.float32)
        self.y  = torch.tensor(y,  dtype=torch.float32)
        self.cc = torch.tensor(cc, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.cc[idx]



# ── Main ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if cache_exists():
        print("Cache found — loading preprocessed sequences...")
        (train_X, train_y, train_cc,
         val_X,   val_y,   val_cc,
         test_X,  test_y,  test_cc, _,
         symbol_scalers, feature_cols, close_col_idx,
         _test_start, _test_end) = load_cache()
        train_ds = CachedDataset(train_X, train_y, train_cc)
        val_ds   = CachedDataset(val_X,   val_y,   val_cc)
        test_ds  = CachedDataset(test_X,  test_y,  test_cc)
        feature_cols_count = train_X.shape[2]
    else:
        print("No cache — running preprocessing (this only happens once)...")
        (train_df, val_df, test_df,
         feature_cols, target_cols,
         symbol_scalers, close_col_idx) = load_and_preprocess(DATA_PATH, TRAIN_RATIO, VAL_RATIO)

        total = len(train_df) + len(val_df) + len(test_df)
        print(f"  {total:,} rows | {len(feature_cols)} features")
        print(f"  Train: {len(train_df):,} ({str(train_df['date'].min())[:10]} -> {str(train_df['date'].max())[:10]})")
        print(f"  Val  : {len(val_df):,} ({str(val_df['date'].min())[:10]} -> {str(val_df['date'].max())[:10]})")
        print(f"  Test : {len(test_df):,} ({str(test_df['date'].min())[:10]} -> {str(test_df['date'].max())[:10]})")

        # Verify targets before sequence building
        sample = train_df.dropna(subset=target_cols).head(3)
        print("\n  Target verification (first 3 rows):")
        print(f"  {'close':>10}  {'target_5d':>10}  {'target_10d':>11}  {'target_20d':>11}")
        for _, row in sample.iterrows():
            print(f"  {row['close']:>10.4f}  {row['target_5d']:>10.4f}"
                  f"  {row['target_10d']:>11.4f}  {row['target_20d']:>11.4f}")
        print("  (values are scaled — should be in [0,1] and target > or < close depending on market)\n")

        print("Building sequence datasets...")
        train_ds           = StockDataset(train_df, feature_cols, target_cols)
        val_ds             = StockDataset(val_df,   feature_cols, target_cols)
        test_ds            = StockDataset(test_df,  feature_cols, target_cols)
        save_cache(train_ds, val_ds, test_ds, symbol_scalers, feature_cols, close_col_idx,
                   test_start=str(test_df["date"].min())[:10],
                   test_end=str(test_df["date"].max())[:10])
        feature_cols_count = len(feature_cols)

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,} sequences")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = LSTMAttentionModel(
        input_size  = feature_cols_count,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
        num_outputs = len(HORIZONS),
    ).to(DEVICE)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters | Device: {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val   = float("inf")
    no_improve = 0
    best_path  = os.path.join(MODEL_DIR, "best_lstm_attention.pt")

    print(f"\nTraining (70% train | 15% val | 15% test) — early stop patience={EARLY_STOP}")
    print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'Dir Acc':>8}  {'':>10}")
    print("-" * 56)

    for epoch in range(1, EPOCHS + 1):
        tr_loss = run_epoch(model, train_loader, optimizer, criterion, training=True)
        va_loss = run_epoch(model, val_loader,   optimizer, criterion, training=False)
        acc     = directional_accuracy(model, val_loader)
        scheduler.step(va_loss)

        marker = ""
        if va_loss < best_val:
            best_val   = va_loss
            no_improve = 0
            torch.save({"model_state":       model.state_dict(),
                        "symbol_scalers":     symbol_scalers,
                        "feature_cols":       feature_cols,
                        "close_col_idx":      close_col_idx,
                        "feature_cols_count": feature_cols_count},
                       best_path)
            marker = "best"
        else:
            no_improve += 1
            marker = f"no imp {no_improve}/{EARLY_STOP}"

        print(f"{epoch:>6d}  {tr_loss:>10.6f}  {va_loss:>10.6f}  {acc:>7.2f}%  {marker}")

        if no_improve >= EARLY_STOP:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest Val MSE: {best_val:.6f}  -> saved to {best_path}")

    # ── Final test evaluation ─────────────────────────────────────────────
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_tgts, all_cc = [], [], []
    with torch.no_grad():
        for X, y, cc in test_loader:
            pred, _ = model(X.to(DEVICE))
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(y.numpy())
            all_cc.append(cc.numpy())

    preds_s = np.concatenate(all_preds)
    tgts_s  = np.concatenate(all_tgts)
    cc_s    = np.concatenate(all_cc)

    print("\n" + "-" * 48)
    print("Final TEST set metrics (MSE in scaled space):")
    print(f"  {'Horizon':>10}  {'RMSE':>8}  {'MAE':>8}  {'Dir Acc':>9}")
    print("  " + "-" * 38)
    for i, h in enumerate(HORIZONS):
        rmse    = np.sqrt(np.mean((preds_s[:, i] - tgts_s[:, i]) ** 2))
        mae     = np.mean(np.abs(preds_s[:, i] - tgts_s[:, i]))
        dir_acc = np.mean((preds_s[:, i] > cc_s) == (tgts_s[:, i] > cc_s)) * 100
        print(f"  {h:>8d}d  {rmse:>8.4f}  {mae:>8.4f}  {dir_acc:>8.2f}%")


if __name__ == "__main__":
    main()
