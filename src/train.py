import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ── Config ───────────────────────────────────────────────────────────────
DATA_PATH   = "data/combined_features.parquet"
MODEL_DIR   = "models"
LOOKBACK    = 60                # days of history per sequence
HORIZONS    = [5, 7, 20]        # predict close price N days ahead
TRAIN_RATIO = 0.80
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DROP_COLS = ["symbol", "date", "company_name", "sector", "industry", "cap_category"]

# ── Data ─────────────────────────────────────────────────────────────────
def load_and_preprocess(path):
    df = pd.read_parquet(path)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = df.dropna(subset=["close"])

    # Forward-fill then back-fill per symbol (linear/mean estimation per paper §3.1)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df[numeric_cols] = df.groupby("symbol")[numeric_cols].transform(lambda x: x.ffill().bfill())

    # Add target columns: close price N days ahead, per symbol
    for h in HORIZONS:
        df[f"target_{h}d"] = df.groupby("symbol")["close"].shift(-h)

    feature_cols = [c for c in df.columns if c not in DROP_COLS and not c.startswith("target_")]
    return df, feature_cols


def chronological_split(df, ratio=0.80):
    dates     = np.sort(df["date"].unique())
    cutoff    = dates[int(len(dates) * ratio)]
    train_df  = df[df["date"] <  cutoff].copy()
    test_df   = df[df["date"] >= cutoff].copy()
    return train_df, test_df


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
    """Additive attention over LSTM hidden states (paper §3.3 eq. 4–5)."""
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden)
        e = self.score(lstm_out).squeeze(-1)          # (batch, seq_len)
        a = torch.softmax(e, dim=1).unsqueeze(-1)     # (batch, seq_len, 1)
        context = (a * lstm_out).sum(dim=1)           # (batch, hidden)
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
        out              = self.dropout(context)
        return self.fc(out), weights


# ── Train / Eval loops ───────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, training):
    model.train() if training else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred, _ = model(X)
            loss    = criterion(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading & preprocessing data...")
    df, feature_cols = load_and_preprocess(DATA_PATH)
    print(f"  {len(df):,} rows | {len(feature_cols)} features | "
          f"symbols: {df['symbol'].nunique()}")

    train_df, test_df = chronological_split(df, TRAIN_RATIO)
    print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    print("Building sequence datasets...")
    train_ds = StockDataset(train_df, feature_cols, fit=True)
    test_ds  = StockDataset(test_df,  feature_cols,
                            feat_scaler=train_ds.feat_scaler,
                            tgt_scaler=train_ds.tgt_scaler)
    print(f"  Train sequences: {len(train_ds):,} | Test sequences: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = LSTMAttentionModel(
        input_size  = len(feature_cols),
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
        num_outputs = len(HORIZONS),
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters | Device: {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val   = float("inf")
    best_path  = os.path.join(MODEL_DIR, "best_lstm_attention.pt")

    print(f"\nTraining for {EPOCHS} epochs...\n" + "-"*55)
    for epoch in range(1, EPOCHS + 1):
        tr_loss = run_epoch(model, train_loader, optimizer, criterion, training=True)
        va_loss = run_epoch(model, test_loader,  optimizer, criterion, training=False)
        scheduler.step(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model_state": model.state_dict(),
                        "feat_scaler": train_ds.feat_scaler,
                        "tgt_scaler":  train_ds.tgt_scaler,
                        "feature_cols": feature_cols},
                       best_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"Train MSE: {tr_loss:.6f}  Val MSE: {va_loss:.6f}"
                  + (" *" if va_loss == best_val else ""))

    print(f"\nBest Val MSE: {best_val:.6f}  →  saved to {best_path}")

    # ── Per-horizon RMSE on original price scale ─────────────────────────
    checkpoint = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_tgts = [], []
    with torch.no_grad():
        for X, y in test_loader:
            pred, _ = model(X.to(DEVICE))
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(y.numpy())

    preds_norm = np.concatenate(all_preds)
    tgts_norm  = np.concatenate(all_tgts)

    tgt_scaler  = checkpoint["tgt_scaler"]
    preds_price = tgt_scaler.inverse_transform(preds_norm)
    tgts_price  = tgt_scaler.inverse_transform(tgts_norm)

    print("\n" + "-"*40)
    print("Test RMSE (original price scale ₹):")
    for i, h in enumerate(HORIZONS):
        rmse = np.sqrt(np.mean((preds_price[:, i] - tgts_price[:, i]) ** 2))
        mae  = np.mean(np.abs(preds_price[:, i] - tgts_price[:, i]))
        print(f"  {h:2d}-day horizon  RMSE: {rmse:8.2f}  MAE: {mae:8.2f}")


if __name__ == "__main__":
    main()
