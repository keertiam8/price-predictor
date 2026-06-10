import os
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR   = "data/cache"
MODEL_PATH  = "models/best_lstm_attention.pt"
HORIZONS    = [5, 10, 20]
BATCH_SIZE  = 128
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def run_validation():
    print(f"Loading model from {MODEL_PATH} ...")
    checkpoint     = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    symbol_scalers = checkpoint["symbol_scalers"]   # {sym_id: {"feat": sc, "tgt": sc}}
    input_size     = checkpoint["feature_cols_count"]

    model = LSTMAttentionModel(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, len(HORIZONS)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"  Device: {DEVICE}")

    print(f"\nLoading test data from {CACHE_DIR} ...")
    test_X       = np.load(f"{CACHE_DIR}/test_X.npy")
    test_y       = np.load(f"{CACHE_DIR}/test_y.npy")
    test_sym_ids = np.load(f"{CACHE_DIR}/test_sym_ids.npy")   # symbol id per sequence
    print(f"  Test sequences: {len(test_X):,}")

    # Run inference in batches
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_X), BATCH_SIZE):
            X_batch = torch.tensor(test_X[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            pred, _ = model(X_batch)
            all_preds.append(pred.cpu().numpy())

    preds_scaled = np.concatenate(all_preds)
    tgts_scaled  = test_y

    # ── Inverse transform per symbol (using saved sym_ids for correct boundaries) ──
    preds_price = np.zeros_like(preds_scaled)
    tgts_price  = np.zeros_like(tgts_scaled)

    for sym_id in sorted(symbol_scalers.keys()):
        mask = test_sym_ids == sym_id
        if mask.sum() == 0:
            continue
        sc = symbol_scalers[sym_id]["tgt"]
        preds_price[mask] = sc.inverse_transform(preds_scaled[mask])
        tgts_price[mask]  = sc.inverse_transform(tgts_scaled[mask])

    # ── Metrics ──────────────────────────────────────────────────────────────
    def horizon_block(h_idx, h):
        p_s  = preds_scaled[:, h_idx]
        t_s  = tgts_scaled[:, h_idx]
        p_r  = preds_price[:, h_idx]
        t_r  = tgts_price[:, h_idx]

        rmse_s = np.sqrt(np.nanmean((p_s - t_s) ** 2))
        mae_s  = np.nanmean(np.abs(p_s - t_s))
        rmse_r = np.sqrt(np.nanmean((p_r - t_r) ** 2))
        mae_r  = np.nanmean(np.abs(p_r - t_r))

        # Targets are log-returns scaled to [0,1]; 0.5 ≈ zero return
        dir_acc = np.mean((p_s > 0.5) == (t_s > 0.5)) * 100

        print(f"\n{'='*60}")
        print(f"  {h}-DAY HORIZON")
        print(f"{'='*60}")
        print(f"  RMSE (scaled [0-1])  : {rmse_s:>10.4f}")
        print(f"  MAE  (scaled [0-1])  : {mae_s:>10.4f}")
        print(f"  RMSE (price space)   : {rmse_r:>10.2f}  (approx, see note)")
        print(f"  MAE  (price space)   : {mae_r:>10.2f}  (approx, see note)")
        print(f"  Directional Acc      : {dir_acc:>9.2f}%")
        print(f"  (Dir: predicted log-return > 0 == actual log-return > 0)")
        print(f"{'─'*60}")
        print(f"  Sample predictions (first 8 test sequences):")
        print(f"  {'#':>4}  {'Actual(r)':>10}  {'Pred(r)':>10}  {'Err(r)':>9}  {'Correct':>7}")
        print(f"  {'─'*52}")
        for j in range(min(8, len(p_r))):
            err     = p_r[j] - t_r[j]
            correct = (p_s[j] > 0.5) == (t_s[j] > 0.5)
            tick    = "YES" if correct else "NO"
            print(f"  {j+1:>4}  {t_r[j]:>10.2f}  {p_r[j]:>10.2f}  {err:>+9.2f}  {tick:>7}")

        return rmse_s, mae_s, rmse_r, mae_r, dir_acc

    all_rmse_s, all_mae_s, all_rmse_r, all_mae_r, all_dir = [], [], [], [], []
    for i, h in enumerate(HORIZONS):
        rs, ms, rr, mr, d = horizon_block(i, h)
        all_rmse_s.append(rs); all_mae_s.append(ms)
        all_rmse_r.append(rr); all_mae_r.append(mr)
        all_dir.append(d)

    print(f"\n{'='*68}")
    print(f"  COMBINED SUMMARY (all horizons)")
    print(f"{'='*68}")
    print(f"  {'Horizon':>10}  {'RMSE(sc)':>9}  {'MAE(sc)':>8}  {'RMSE(rs)':>9}  {'MAE(rs)':>8}  {'Dir Acc':>8}")
    print(f"  {'─'*62}")
    for i, h in enumerate(HORIZONS):
        print(f"  {h:>8d}d  {all_rmse_s[i]:>9.4f}  {all_mae_s[i]:>8.4f}"
              f"  {all_rmse_r[i]:>9.2f}  {all_mae_r[i]:>8.2f}  {all_dir[i]:>7.2f}%")
    print(f"  {'─'*62}")
    print(f"  {'Average':>10}  {np.mean(all_rmse_s):>9.4f}  {np.mean(all_mae_s):>8.4f}"
          f"  {np.mean(all_rmse_r):>9.2f}  {np.mean(all_mae_r):>8.2f}  {np.mean(all_dir):>7.2f}%")
    print(f"{'='*68}")
    print(f"\n  NOTE: Price-space RMSE uses per-symbol MinMaxScaler trained on 1997-2017.")
    print(f"  Stocks that grew beyond their training range (e.g. BAJFINANCE, RELIANCE)")
    print(f"  will show large price errors. Scaled-space RMSE and Dir Acc are more reliable.")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"No trained model at {MODEL_PATH} — run train.py first.")
    elif not os.path.exists(f"{CACHE_DIR}/test_X.npy"):
        print(f"No cache at {CACHE_DIR} — run train.py first.")
    elif not os.path.exists(f"{CACHE_DIR}/test_sym_ids.npy"):
        print(f"Cache is outdated — delete data/cache/ and rerun train.py.")
    else:
        run_validation()
