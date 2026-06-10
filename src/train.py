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
VAL_RATIO    = 0.15          # test gets the remaining 0.15
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

    # Label encode — fit on full data so all symbols/sectors are covered
    encoders = {}
    for col in ["symbol", "sector", "cap_category"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  Encoded {col}: {dict(enumerate(le.classes_))}")

    # ── Split FIRST, then fill — prevents bfill leakage ──────────────────
    dates     = np.sort(df["date"].unique())
    train_cut = dates[int(len(dates) * train_ratio)]
    val_cut   = dates[int(len(dates) * (train_ratio + val_ratio))]

    train_df = df[df["date"] <  train_cut].copy()
    val_df   = df[(df["date"] >= train_cut) & (df["date"] < val_cut)].copy()
    test_df  = df[df["date"] >= val_cut].copy()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Train: ffill then bfill (bfill only fills NaNs at the very start of history)
    train_df[numeric_cols] = train_df.groupby("symbol")[numeric_cols].transform(
        lambda x: x.ffill().bfill()
    )

    # Val/Test: ffill only — no looking into the future
    val_df[numeric_cols]  = val_df.groupby("symbol")[numeric_cols].transform(lambda x: x.ffill())
    test_df[numeric_cols] = test_df.groupby("symbol")[numeric_cols].transform(lambda x: x.ffill())

    # Add targets after split (so shift is within each split's symbol group)
    for split in [train_df, val_df, test_df]:
        for h in HORIZONS:
            split[f"target_{h}d"] = split.groupby("symbol")["close"].shift(-h)

    feature_cols = [c for c in df.columns if c not in DROP_COLS and not c.startswith("target_")]
    return train_df, val_df, test_df, feature_cols, encoders


def chronological_split(df, train_ratio=0.70, val_ratio=0.15):
    dates     = np.sort(df["date"].unique())
    train_cut = dates[int(len(dates) * train_ratio)]
    val_cut   = dates[int(len(dates) * (train_ratio + val_ratio))]
    train_df  = df[df["date"] <  train_cut].copy()
    val_df    = df[(df["date"] >= train_cut) & (df["date"] < val_cut)].copy()
    test_df   = df[df["date"] >= val_cut].copy()
    return train_df, val_df, test_df


class StockDataset(Dataset):
    def __init__(self, df, feature_cols, feat_scaler=None, tgt_scaler=None, fit=False):
        target_cols = [f"target_{h}d" for h in HORIZONS]
        df = df.dropna(subset=target_cols).copy()

        feat_vals = df[feature_cols].values.astype(np.float32)
        tgt_vals  = df[target_cols].values.astype(np.float32)

        if fit:
            self.feat_scaler = MinMaxScaler()
            self.tgt_scaler  = MinMaxScaler()
            feat_vals = self.feat_scaler.fit_transform(feat_vals)
            tgt_vals  = self.tgt_scaler.fit_transform(tgt_vals)
        else:
            self.feat_scaler = feat_scaler
            self.tgt_scaler  = tgt_scaler
            feat_vals = self.feat_scaler.transform(feat_vals)
            tgt_vals  = self.tgt_scaler.transform(tgt_vals)

        feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=1.0, neginf=0.0)
        tgt_vals  = np.nan_to_num(tgt_vals,  nan=0.0, posinf=1.0, neginf=0.0)

        df = df.copy()
        df[feature_cols] = feat_vals
        df[target_cols]  = tgt_vals

        self.sequences, self.targets = [], []
        for _, grp in df.groupby("symbol"):
            grp   = grp.reset_index(drop=True)
            feats = grp[feature_cols].values.astype(np.float32)
            tgts  = grp[target_cols].values.astype(np.float32)
            for i in range(LOOKBACK, len(grp)):
                self.sequences.append(feats[i - LOOKBACK : i])
                self.targets.append(tgts[i])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets   = np.array(self.targets,   dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])


# ── Model ─────────────────────────────────────────────────────────────────
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        e = self.score(lstm_out).squeeze(-1)
        a = torch.softmax(e, dim=1).unsqueeze(-1)
        context = (a * lstm_out).sum(dim=1)
        return context, a.squeeze(-1)


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
        for X, y in loader:
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
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            pred, _ = model(X.to(DEVICE))
            pred = pred.cpu()
            current    = X[:, -1, 5].unsqueeze(1)
            pred_dir   = (pred > current).float()
            actual_dir = (y    > current).float()
            correct += (pred_dir == actual_dir).all(dim=1).sum().item()
            total   += len(y)
    return 100.0 * correct / total if total > 0 else 0.0


# ── Cache ─────────────────────────────────────────────────────────────────
def save_cache(train_ds, val_ds, test_ds):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(f"{CACHE_DIR}/train_X.npy", train_ds.sequences)
    np.save(f"{CACHE_DIR}/train_y.npy", train_ds.targets)
    np.save(f"{CACHE_DIR}/val_X.npy",   val_ds.sequences)
    np.save(f"{CACHE_DIR}/val_y.npy",   val_ds.targets)
    np.save(f"{CACHE_DIR}/test_X.npy",  test_ds.sequences)
    np.save(f"{CACHE_DIR}/test_y.npy",  test_ds.targets)
    with open(f"{CACHE_DIR}/scalers.pkl", "wb") as f:
        pickle.dump({"feat_scaler": train_ds.feat_scaler,
                     "tgt_scaler":  train_ds.tgt_scaler}, f)
    print("  Cache saved to", CACHE_DIR)


def cache_exists():
    files = ["train_X.npy", "train_y.npy", "val_X.npy", "val_y.npy",
             "test_X.npy",  "test_y.npy",  "scalers.pkl"]
    return all(os.path.exists(f"{CACHE_DIR}/{f}") for f in files)


def load_cache():
    train_X = np.load(f"{CACHE_DIR}/train_X.npy")
    train_y = np.load(f"{CACHE_DIR}/train_y.npy")
    val_X   = np.load(f"{CACHE_DIR}/val_X.npy")
    val_y   = np.load(f"{CACHE_DIR}/val_y.npy")
    test_X  = np.load(f"{CACHE_DIR}/test_X.npy")
    test_y  = np.load(f"{CACHE_DIR}/test_y.npy")
    with open(f"{CACHE_DIR}/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    return train_X, train_y, val_X, val_y, test_X, test_y, \
           scalers["feat_scaler"], scalers["tgt_scaler"]


class CachedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if cache_exists():
        print("Cache found — loading preprocessed sequences...")
        train_X, train_y, val_X, val_y, test_X, test_y, feat_scaler, tgt_scaler = load_cache()
        train_ds = CachedDataset(train_X, train_y)
        val_ds   = CachedDataset(val_X,   val_y)
        test_ds  = CachedDataset(test_X,  test_y)
        feature_cols_count = train_X.shape[2]
    else:
        print("No cache — running preprocessing (this only happens once)...")
        train_df, val_df, test_df, feature_cols, _ = load_and_preprocess(DATA_PATH, TRAIN_RATIO, VAL_RATIO)
        total = len(train_df) + len(val_df) + len(test_df)
        print(f"  {total:,} rows | {len(feature_cols)} features")
        print(f"  Train: {len(train_df):,} ({str(train_df['date'].min())[:10]} -> {str(train_df['date'].max())[:10]})")
        print(f"  Val  : {len(val_df):,} ({str(val_df['date'].min())[:10]} -> {str(val_df['date'].max())[:10]})")
        print(f"  Test : {len(test_df):,} ({str(test_df['date'].min())[:10]} -> {str(test_df['date'].max())[:10]})")

        print("Building sequence datasets...")
        train_ds = StockDataset(train_df, feature_cols, fit=True)
        val_ds   = StockDataset(val_df,   feature_cols,
                                feat_scaler=train_ds.feat_scaler,
                                tgt_scaler=train_ds.tgt_scaler)
        test_ds  = StockDataset(test_df,  feature_cols,
                                feat_scaler=train_ds.feat_scaler,
                                tgt_scaler=train_ds.tgt_scaler)
        save_cache(train_ds, val_ds, test_ds)
        feat_scaler        = train_ds.feat_scaler
        tgt_scaler         = train_ds.tgt_scaler
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
                        "feat_scaler":        feat_scaler,
                        "tgt_scaler":         tgt_scaler,
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

    # ── Final evaluation on held-out TEST set ────────────────────────────
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_tgts = [], []
    with torch.no_grad():
        for X, y in test_loader:
            pred, _ = model(X.to(DEVICE))
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(y.numpy())

    preds_price = tgt_scaler.inverse_transform(np.concatenate(all_preds))
    tgts_price  = tgt_scaler.inverse_transform(np.concatenate(all_tgts))

    print("\n" + "-" * 48)
    print("Final TEST set metrics (never seen during training):")
    print(f"  {'Horizon':>10}  {'RMSE':>8}  {'MAE':>8}  {'MAPE':>8}")
    print("  " + "-" * 38)
    for i, h in enumerate(HORIZONS):
        rmse = np.sqrt(np.mean((preds_price[:, i] - tgts_price[:, i]) ** 2))
        mae  = np.mean(np.abs(preds_price[:, i]  - tgts_price[:, i]))
        mape = np.mean(np.abs((preds_price[:, i] - tgts_price[:, i]) / (tgts_price[:, i] + 1e-8))) * 100
        print(f"  {h:>8d}d  {rmse:>8.2f}  {mae:>8.2f}  {mape:>7.2f}%")


if __name__ == "__main__":
    main()
