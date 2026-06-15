import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH        = "data/combined_features.parquet"
MODEL_DIR        = "models"
CACHE_DIR        = "data/cache"
LOOKBACK         = 60
HORIZONS         = [5, 10, 20]
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
TRAIN_START_DATE = "2014-01-01"
BATCH_SIZE       = 64
EPOCHS           = 100
LR               = 1e-3
HIDDEN_SIZE      = 256
NUM_LAYERS       = 3
DROPOUT          = 0.2
EARLY_STOP       = 25
WEIGHT_DECAY     = 1e-4
DIR_WEIGHT       = 1.0   # BCE direction loss weight vs Huber magnitude loss
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DROP_COLS = ["date", "company_name"]

_OHLCV_PRICE = ["open", "high", "low", "close", "volume"]
_MA_COLS     = ["50d_ma", "200d_ma", "20d_avg_volume"]
_MACRO_LEVEL = ["nifty50", "usd_inr", "brent_crude_usd", "wti_crude_usd"]

_DEMERGER_DATES = {
    ("BAJFINANCE", "2024-07-08"),
    ("HDFCBANK",   "2024-07-08"),
    ("RELIANCE",   "2024-07-08"),
}


# ── Preprocessing ─────────────────────────────────────────────────────────
def _transform_to_returns(df):
    df = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)

    for sym_id, idx in df.groupby("symbol").groups.items():
        g          = df.loc[idx].sort_values("date")
        close      = g["close"]
        prev_close = close.shift(1)

        df.loc[g.index, "log_return"]  = np.log((close / prev_close).clip(1e-9))
        df.loc[g.index, "open_return"] = np.log((g["open"] / prev_close).clip(1e-9))
        df.loc[g.index, "high_ret"]    = np.log((g["high"] / close).clip(1e-9))
        df.loc[g.index, "low_ret"]     = np.log((g["low"]  / close).clip(1e-9))
        df.loc[g.index, "volume_chg"]  = g["volume"].pct_change(fill_method=None).clip(-10, 10)

        sym_str    = df.loc[g.index[0], "symbol"] if isinstance(sym_id, int) else str(sym_id)
        split_mask = g.get("is_split", pd.Series(0, index=g.index)).astype(bool)
        df.loc[g.index[split_mask.values], "log_return"]  = 0.0
        df.loc[g.index[split_mask.values], "open_return"] = 0.0
        for date_str in [d for s, d in _DEMERGER_DATES if s == sym_str]:
            dmask = g["date"].astype(str).str[:10] == date_str
            df.loc[g.index[dmask.values], "log_return"]  = 0.0
            df.loc[g.index[dmask.values], "open_return"] = 0.0

        for ma_col, new_col in [("50d_ma", "ma50_dev"), ("200d_ma", "ma200_dev")]:
            if ma_col in df.columns:
                df.loc[g.index, new_col] = (close / g[ma_col].replace(0, np.nan) - 1).clip(-2, 2)
        if "20d_avg_volume" in df.columns:
            df.loc[g.index, "vol_ratio_dev"] = (
                g["volume"] / g["20d_avg_volume"].replace(0, np.nan) - 1
            ).clip(-10, 10)

    for col in _MACRO_LEVEL:
        if col in df.columns:
            df[f"{col}_chg"] = df.groupby("symbol")[col].pct_change(fill_method=None).clip(-2, 2)

    drop_orig = _OHLCV_PRICE + _MA_COLS + _MACRO_LEVEL
    df = df.drop(columns=[c for c in drop_orig if c in df.columns], errors="ignore")
    return df


def load_and_preprocess(path, train_ratio=0.70, val_ratio=0.15):
    df = pd.read_parquet(path)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = df.dropna(subset=["close"])

    for col in ["symbol", "sector"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Encoded {col}: {dict(enumerate(le.classes_))}")

    raw_close = df[["symbol", "date", "close"]].copy()
    df = _transform_to_returns(df)
    df = df.merge(raw_close.rename(columns={"close": "_raw_close"}),
                  on=["symbol", "date"], how="left")

    n_before = len(df)
    df = df[df["date"] >= pd.Timestamp(TRAIN_START_DATE)].reset_index(drop=True)
    print(f"  Regime filter: kept {len(df):,}/{n_before:,} rows from {TRAIN_START_DATE} onward")

    dates     = np.sort(df["date"].unique())
    train_cut = dates[int(len(dates) * train_ratio)]
    val_cut   = dates[int(len(dates) * (train_ratio + val_ratio))]

    train_df = df[df["date"] <  train_cut].copy()
    val_df   = df[(df["date"] >= train_cut) & (df["date"] < val_cut)].copy()
    test_df  = df[df["date"] >= val_cut].copy()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for split_df in (train_df, val_df, test_df):
        split_df[numeric_cols] = split_df.groupby("symbol")[numeric_cols].transform(
            lambda x: x.ffill().bfill()
        )

    target_cols  = [f"target_{h}d" for h in HORIZONS]
    feature_cols = [c for c in df.columns
                    if c not in DROP_COLS + ["_raw_close"]
                    and not c.startswith("target_")]

    all_nan_cols = [c for c in feature_cols if train_df[c].isna().all()]
    if all_nan_cols:
        print(f"  Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]

    for split in [train_df, val_df, test_df]:
        for h in HORIZONS:
            future_close = split.groupby("symbol")["_raw_close"].shift(-h)
            cur_close    = split["_raw_close"].replace(0, np.nan)
            split[f"target_{h}d"] = np.log((future_close / cur_close).clip(1e-9))

    target_scalers = {}
    for h in HORIZONS:
        col        = f"target_{h}d"
        sc         = StandardScaler()
        train_vals = train_df[col].dropna().values.reshape(-1, 1)
        sc.fit(train_vals)
        target_scalers[h] = sc
        for split_df in [train_df, val_df, test_df]:
            mask = split_df[col].notna()
            vals = split_df.loc[mask, col].values.reshape(-1, 1).astype(np.float32)
            split_df.loc[mask, col] = sc.transform(vals).ravel()

    symbol_scalers = {}
    for sym_id in sorted(train_df["symbol"].unique()):
        sym_train = train_df[train_df["symbol"] == sym_id][feature_cols].values.astype(np.float32)
        feat_sc   = MinMaxScaler(clip=True)
        feat_sc.fit(np.nan_to_num(sym_train, nan=0.0))
        symbol_scalers[sym_id] = {"feat": feat_sc}

    for split_df in [train_df, val_df, test_df]:
        for sym_id, sc in symbol_scalers.items():
            mask = split_df["symbol"] == sym_id
            if mask.sum() == 0:
                continue
            fv     = split_df.loc[mask, feature_cols].values.astype(np.float32)
            fv     = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
            scaled = sc["feat"].transform(fv)
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
            split_df.loc[mask, feature_cols] = scaled

    log_return_col_idx = feature_cols.index("log_return") if "log_return" in feature_cols else 0

    total = len(train_df) + len(val_df) + len(test_df)
    print(f"  {total:,} rows | {len(feature_cols)} features")
    print(f"  Train: {len(train_df):,} ({str(train_df['date'].min())[:10]} -> {str(train_df['date'].max())[:10]})")
    print(f"  Val  : {len(val_df):,} ({str(val_df['date'].min())[:10]} -> {str(val_df['date'].max())[:10]})")
    print(f"  Test : {len(test_df):,} ({str(test_df['date'].min())[:10]} -> {str(test_df['date'].max())[:10]})")

    print("\n  Class balance (% UP) per split:")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        row = f"    {name:>6}: "
        for h in HORIZONS:
            vals = split[f"target_{h}d"].dropna()
            pct_up = (vals > float(target_scalers[h].transform([[0.0]])[0, 0])).mean() * 100
            row += f"  {h}d={pct_up:.1f}%UP"
        print(row)

    return (train_df, val_df, test_df, feature_cols, target_cols,
            symbol_scalers, target_scalers, log_return_col_idx)


# ── Dataset ────────────────────────────────────────────────────────────────
class StockDataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, log_return_col_idx=0):
        df = df.dropna(subset=target_cols).copy()
        self.sequences, self.targets, self.current_closes = [], [], []

        for _, grp in df.groupby("symbol"):
            grp   = grp.reset_index(drop=True)
            feats = grp[feature_cols].values.astype(np.float32)
            tgts  = grp[target_cols].values.astype(np.float32)
            for i in range(LOOKBACK, len(grp)):
                self.sequences.append(feats[i - LOOKBACK : i])
                self.targets.append(tgts[i])
                self.current_closes.append(feats[i, log_return_col_idx])

        self.sequences      = np.array(self.sequences,      dtype=np.float32)
        self.targets        = np.array(self.targets,        dtype=np.float32)
        self.current_closes = np.array(self.current_closes, dtype=np.float32)

        n_nan = int(np.isnan(self.sequences).sum())
        if n_nan:
            print(f"    [StockDataset] zeroed {n_nan:,} residual NaN feature cells")
        self.sequences = np.nan_to_num(self.sequences, nan=0.0, posinf=1.0, neginf=0.0)

    def __len__(self):  return len(self.sequences)
    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx]),
                torch.tensor(self.targets[idx]),
                torch.tensor(self.current_closes[idx]))


# ── Model ──────────────────────────────────────────────────────────────────
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        e = self.score(lstm_out).squeeze(-1)
        a = torch.softmax(e, dim=1).unsqueeze(-1)
        return (a * lstm_out).sum(dim=1), a.squeeze(-1)


class TwoStageModel(nn.Module):
    """Two-stage LSTM+attention model.

    Stage 1 — Direction Classifier (cls_head):
        Predicts P(return > 0) as logits for each horizon.
        Loss: BCEWithLogitsLoss.

    Stage 2 — Magnitude Regressor (reg_head):
        Predicts |return| magnitude using the shared LSTM context
        concatenated with detached Stage 1 probabilities.
        Loss: HuberLoss.

    Final prediction: sign(Stage 1 logit) × relu(Stage 2 magnitude).
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_horizons):
        super().__init__()
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0.0)
        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.cls_head  = nn.Linear(hidden_size, num_horizons)
        # Stage 2 takes shared context + detached Stage 1 probabilities
        self.reg_head  = nn.Linear(hidden_size + num_horizons, num_horizons)

    def forward(self, x):
        lstm_out, _      = self.lstm(x)
        context, weights = self.attention(lstm_out)
        z = self.dropout(context)

        cls_logits = self.cls_head(z)                          # (B, H) direction logits
        cls_probs  = torch.sigmoid(cls_logits).detach()        # (B, H) — no gradient back to Stage 1
        mag_preds  = F.relu(self.reg_head(torch.cat([z, cls_probs], dim=-1)))  # (B, H) non-negative

        return cls_logits, mag_preds, weights


# ── Loss ───────────────────────────────────────────────────────────────────
def two_stage_loss(cls_logits, mag_preds, y_std, zero_thresh, dir_weight=1.0, pos_weight=None):
    """BCE for direction (Stage 1) + Huber for magnitude (Stage 2).

    Targets derived from standardised log-returns:
      y_dir = 1 if raw return > 0  (threshold is zero_thresh in std space)
      y_mag = |y_std - zero_thresh| = distance from zero-return in std space

    pos_weight (per-horizon, = #down/#up in train) down-weights the UP class so
    the classifier cannot trivially win by always predicting the majority (UP)
    direction — it must earn its UP calls. Without it the classifier collapses
    to ~90% UP and never beats the always-UP baseline.
    """
    zt    = zero_thresh.unsqueeze(0)           # (1, H) → broadcasts with (B, H)
    y_dir = (y_std > zt).float()
    y_mag = torch.abs(y_std - zt)

    cls_loss = F.binary_cross_entropy_with_logits(cls_logits, y_dir, pos_weight=pos_weight)
    reg_loss = F.huber_loss(mag_preds, y_mag, delta=1.0)
    total    = dir_weight * cls_loss + reg_loss
    return total, cls_loss.item(), reg_loss.item()


# ── Train / Eval ───────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, zero_thresh, dir_weight, pos_weight, training):
    model.train() if training else model.eval()
    total_loss = cls_total = reg_total = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, y, _ in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            cls_logits, mag_preds, _ = model(X)
            loss, c_l, r_l = two_stage_loss(cls_logits, mag_preds, y,
                                             zero_thresh, dir_weight, pos_weight)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n = len(X)
            total_loss += loss.item() * n
            cls_total  += c_l * n
            reg_total  += r_l * n
    N = len(loader.dataset)
    return total_loss / N, cls_total / N, reg_total / N


def directional_accuracy(model, loader, zero_thresh):
    """Returns (avg_skill, avg_acc, per_horizon_acc, per_horizon_baseline).

    skill = model_dir_acc - always_UP_baseline for each horizon.
    Checkpointing on avg_skill means we only save when the model
    genuinely outperforms the trivial majority-class predictor.
    """
    model.eval()
    correct  = torch.zeros(len(HORIZONS))
    baseline = torch.zeros(len(HORIZONS))  # always-UP correct = fraction of UP labels
    total    = 0
    zt       = zero_thresh.cpu().unsqueeze(0)
    with torch.no_grad():
        for X, y, _ in loader:
            cls_logits, _, _ = model(X.to(DEVICE))
            cls_logits = cls_logits.cpu()
            y_up = (y > zt)
            correct  += ((cls_logits > 0) == y_up).float().sum(dim=0)
            baseline += y_up.float().sum(dim=0)   # always-UP gets this many right
            total    += len(y)
    per_acc      = (correct  / total * 100).tolist()
    per_baseline = (baseline / total * 100).tolist()
    per_skill    = [a - b for a, b in zip(per_acc, per_baseline)]
    avg_skill    = float(np.mean(per_skill))
    avg_acc      = float(np.mean(per_acc))
    return avg_skill, avg_acc, per_acc, per_baseline


# ── Cache ──────────────────────────────────────────────────────────────────
def save_cache(train_ds, val_ds, test_ds, symbol_scalers, target_scalers,
               feature_cols, close_col_idx, test_start=None, test_end=None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(f"{CACHE_DIR}/train_X.npy",  train_ds.sequences)
    np.save(f"{CACHE_DIR}/train_y.npy",  train_ds.targets)
    np.save(f"{CACHE_DIR}/train_cc.npy", train_ds.current_closes)
    np.save(f"{CACHE_DIR}/val_X.npy",    val_ds.sequences)
    np.save(f"{CACHE_DIR}/val_y.npy",    val_ds.targets)
    np.save(f"{CACHE_DIR}/val_cc.npy",   val_ds.current_closes)
    np.save(f"{CACHE_DIR}/test_X.npy",   test_ds.sequences)
    np.save(f"{CACHE_DIR}/test_y.npy",   test_ds.targets)
    np.save(f"{CACHE_DIR}/test_cc.npy",  test_ds.current_closes)
    with open(f"{CACHE_DIR}/meta.pkl", "wb") as f:
        pickle.dump({"symbol_scalers":  symbol_scalers,
                     "target_scalers":  target_scalers,
                     "feature_cols":    feature_cols,
                     "close_col_idx":   close_col_idx,
                     "test_start_date": str(test_start),
                     "test_end_date":   str(test_end)}, f)
    print("  Cache saved to", CACHE_DIR)


def cache_exists():
    files = ["train_X.npy", "train_y.npy", "train_cc.npy",
             "val_X.npy",   "val_y.npy",   "val_cc.npy",
             "test_X.npy",  "test_y.npy",  "test_cc.npy", "meta.pkl"]
    return all(os.path.exists(f"{CACHE_DIR}/{f}") for f in files)


def load_cache():
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
    return (train_X, train_y, train_cc,
            val_X,   val_y,   val_cc,
            test_X,  test_y,  test_cc,
            meta["symbol_scalers"], meta.get("target_scalers", {}),
            meta["feature_cols"],   meta["close_col_idx"],
            meta.get("test_start_date"), meta.get("test_end_date"))


class CachedDataset(Dataset):
    def __init__(self, X, y, cc):
        self.X  = torch.tensor(X,  dtype=torch.float32)
        self.y  = torch.tensor(y,  dtype=torch.float32)
        self.cc = torch.tensor(cc, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.cc[idx]


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if cache_exists():
        print("Cache found — loading preprocessed sequences...")
        (train_X, train_y, train_cc,
         val_X,   val_y,   val_cc,
         test_X,  test_y,  test_cc,
         symbol_scalers, target_scalers, feature_cols, log_return_col_idx,
         _test_start, _test_end) = load_cache()
        train_ds = CachedDataset(train_X, train_y, train_cc)
        val_ds   = CachedDataset(val_X,   val_y,   val_cc)
        test_ds  = CachedDataset(test_X,  test_y,  test_cc)
        feature_cols_count = train_X.shape[2]
    else:
        print("No cache — running preprocessing (this only happens once)...")
        (train_df, val_df, test_df,
         feature_cols, target_cols,
         symbol_scalers, target_scalers,
         log_return_col_idx) = load_and_preprocess(DATA_PATH, TRAIN_RATIO, VAL_RATIO)

        print("\nBuilding sequence datasets...")
        train_ds = StockDataset(train_df, feature_cols, target_cols, log_return_col_idx)
        val_ds   = StockDataset(val_df,   feature_cols, target_cols, log_return_col_idx)
        test_ds  = StockDataset(test_df,  feature_cols, target_cols, log_return_col_idx)
        save_cache(train_ds, val_ds, test_ds, symbol_scalers, target_scalers,
                   feature_cols, log_return_col_idx,
                   test_start=str(test_df["date"].min())[:10],
                   test_end=str(test_df["date"].max())[:10])
        feature_cols_count = len(feature_cols)

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,} sequences")

    # zero_thresh: standardised value of raw return = 0 for each horizon
    zero_thresh = torch.tensor(
        [float(target_scalers[h].transform([[0.0]])[0, 0]) for h in HORIZONS],
        dtype=torch.float32, device=DEVICE,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TwoStageModel(
        input_size   = feature_cols_count,
        hidden_size  = HIDDEN_SIZE,
        num_layers   = NUM_LAYERS,
        dropout      = DROPOUT,
        num_horizons = len(HORIZONS),
    ).to(DEVICE)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters | Device: {DEVICE}")

    # pos_weight < 1 down-weights UP so BCE can't win by always predicting majority.
    # n_down/n_up ≈ 0.87 was too weak to escape always-UP; use 0.35 to force real
    # gradient pressure toward DOWN while keeping chronological training order.
    zt_cpu    = zero_thresh.cpu()
    _raw_y    = train_ds.y if hasattr(train_ds, 'y') else torch.tensor(train_ds.targets)
    train_y_t = _raw_y if isinstance(_raw_y, torch.Tensor) else torch.tensor(_raw_y, dtype=torch.float32)
    n_up   = (train_y_t > zt_cpu.unsqueeze(0)).float().sum(dim=0).clamp(min=1)
    n_down = (train_y_t <= zt_cpu.unsqueeze(0)).float().sum(dim=0).clamp(min=1)
    # Aggressive down-weighting: effective UP contribution ≈ 0.35 × n_up / (n_up + n_down)
    pos_weight = ((n_down / n_up) * 0.4).to(DEVICE)
    print(f"  pos_weight (aggressive 0.4×down/up): "
          + " / ".join(f"{h}d={v:.3f}" for h, v in zip(HORIZONS, pos_weight.tolist())))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_skill = -999.0   # skill = val dir acc − always-UP baseline
    no_improve     = 0
    best_path      = os.path.join(MODEL_DIR, "best_lstm_attention.pt")

    print(f"\nTraining — early stop patience={EARLY_STOP}  |  checkpoint on val SKILL (acc − baseline)")
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'ClsLoss':>8}  {'RegLoss':>8}"
          f"  {'ValAcc':>8}  {'Baseline':>9}  {'Skill':>7}  {'5d/10d/20d':>16}")
    print("-" * 92)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_cls, tr_reg = run_epoch(model, train_loader, optimizer,
                                            zero_thresh, DIR_WEIGHT, pos_weight, training=True)
        va_loss, _,      _      = run_epoch(model, val_loader,   optimizer,
                                            zero_thresh, DIR_WEIGHT, pos_weight, training=False)
        skill, acc_avg, acc_per, base_per = directional_accuracy(model, val_loader, zero_thresh)
        scheduler.step(va_loss)

        marker = ""
        if skill > best_val_skill:
            best_val_skill = skill
            no_improve     = 0
            torch.save({"model_type":         "two_stage",
                        "model_state":        model.state_dict(),
                        "symbol_scalers":     symbol_scalers,
                        "target_scalers":     target_scalers,
                        "feature_cols":       feature_cols,
                        "close_col_idx":      log_return_col_idx,
                        "feature_cols_count": feature_cols_count,
                        "hidden_size":        HIDDEN_SIZE,
                        "num_layers":         NUM_LAYERS,
                        "dropout":            DROPOUT},
                       best_path)
            marker = " *"
        else:
            no_improve += 1

        base_avg = float(np.mean(base_per))
        per_str  = "/".join(f"{a:.1f}" for a in acc_per)
        print(f"{epoch:>6d}  {tr_loss:>10.6f}  {tr_cls:>8.5f}  {tr_reg:>8.5f}"
              f"  {acc_avg:>7.2f}%  {base_avg:>8.2f}%  {skill:>+6.2f}%  {per_str:>16}{marker}")

        if no_improve >= EARLY_STOP:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest Val Skill: {best_val_skill:+.2f}%  -> saved to {best_path}")

    # ── Final test evaluation ───────────────────────────────────────────────
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_cls, all_mag, all_tgts = [], [], []
    with torch.no_grad():
        for X, y, _ in test_loader:
            cls_logits, mag_preds, _ = model(X.to(DEVICE))
            all_cls.append(cls_logits.cpu().numpy())
            all_mag.append(mag_preds.cpu().numpy())
            all_tgts.append(y.numpy())

    cls_arr  = np.concatenate(all_cls)   # (N, H) direction logits
    mag_arr  = np.concatenate(all_mag)   # (N, H) magnitude predictions
    tgts_arr = np.concatenate(all_tgts)  # (N, H) standardised targets

    zt_np = np.array([float(target_scalers[h].transform([[0.0]])[0, 0]) for h in HORIZONS])

    def invert(arr_std, h_idx):
        h = HORIZONS[h_idx]
        return target_scalers[h].inverse_transform(arr_std.reshape(-1, 1)).ravel()

    # ── Stage 1: Direction Classifier ──────────────────────────────────────
    print("\n" + "=" * 74)
    print("  STAGE 1 — DIRECTION CLASSIFIER")
    print("=" * 74)
    print(f"  {'Horizon':>10}  {'Dir Acc':>9}  {'Baseline':>9}  {'Skill':>7}"
          f"  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("  " + "-" * 68)
    for i, h in enumerate(HORIZONS):
        t_raw    = invert(tgts_arr[:, i], i)
        y_true   = (t_raw > 0).astype(int)
        y_pred   = (cls_arr[:, i] > 0).astype(int)
        acc      = np.mean(y_true == y_pred) * 100
        baseline = y_true.mean() * 100          # always-UP baseline on test
        skill    = acc - baseline
        prec     = precision_score(y_true, y_pred, zero_division=0) * 100
        rec      = recall_score(y_true, y_pred, zero_division=0) * 100
        f1       = f1_score(y_true, y_pred, zero_division=0) * 100
        print(f"  {h:>8d}d  {acc:>8.2f}%  {baseline:>8.2f}%  {skill:>+6.2f}%"
              f"  {prec:>9.2f}%  {rec:>7.2f}%  {f1:>7.2f}%")

    # ── Stage 2: Magnitude Regressor ───────────────────────────────────────
    print("\n" + "=" * 62)
    print("  STAGE 2 — MAGNITUDE REGRESSOR  (std space)")
    print("=" * 62)
    print(f"  {'Horizon':>10}  {'RMSE(mag)':>10}  {'MAE(mag)':>10}  {'MagMean':>9}  {'MagStd':>8}")
    print("  " + "-" * 52)
    for i, h in enumerate(HORIZONS):
        y_mag_true = np.abs(tgts_arr[:, i] - zt_np[i])
        y_mag_pred = mag_arr[:, i]
        rmse = np.sqrt(np.nanmean((y_mag_pred - y_mag_true) ** 2))
        mae  = np.nanmean(np.abs(y_mag_pred - y_mag_true))
        print(f"  {h:>8d}d  {rmse:>10.4f}  {mae:>10.4f}  {y_mag_pred.mean():>+9.4f}"
              f"  {y_mag_pred.std():>8.4f}")

    # ── Combined: sign(Stage1) × magnitude(Stage2) ─────────────────────────
    print("\n" + "=" * 74)
    print("  COMBINED PREDICTION  sign(cls) × magnitude")
    print("=" * 74)
    print(f"  {'Horizon':>10}  {'RMSE(ret)':>10}  {'MAE(ret)':>9}"
          f"  {'Dir Acc':>9}  {'Baseline':>9}  {'Skill':>7}")
    print("  " + "-" * 62)
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
        print(f"  {h:>8d}d  {rmse:>10.4f}  {mae:>9.4f}"
              f"  {dir_acc:>8.2f}%  {baseline:>8.2f}%  {skill:>+6.2f}%")

    # ── Collapse check ─────────────────────────────────────────────────────
    print(f"\n  COLLAPSE CHECK  (magnitude std should be >0.3; %UP should be 40-60%)")
    print(f"  {'Horizon':>10}  {'MagMean':>9}  {'MagStd':>8}  {'clsUP%':>8}  {'status':>12}")
    print(f"  {'─' * 56}")
    for i, h in enumerate(HORIZONS):
        mag   = mag_arr[:, i]
        mn    = mag.mean()
        std   = mag.std()
        up_pct = (cls_arr[:, i] > 0).mean() * 100
        status = "COLLAPSED" if std < 0.3 else ("biased" if up_pct < 35 or up_pct > 65 else "OK")
        print(f"  {h:>8d}d  {mn:>+9.4f}  {std:>8.4f}  {up_pct:>7.1f}%  {status:>12}")


if __name__ == "__main__":
    main()
