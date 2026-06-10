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
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    input_size = checkpoint["feature_cols_count"]

    model = LSTMAttentionModel(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, len(HORIZONS)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"  Device: {DEVICE}")

    print(f"\nLoading test data from {CACHE_DIR} ...")
    test_X = np.load(f"{CACHE_DIR}/test_X.npy")
    test_y = np.load(f"{CACHE_DIR}/test_y.npy")
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

    # preds_scaled / tgts_scaled are raw log-returns (no target scaler)
    # positive = stock went up, negative = stock went down

    def horizon_block(h_idx, h):
        p = preds_scaled[:, h_idx]   # predicted log-return
        t = tgts_scaled[:, h_idx]    # actual log-return

        rmse    = np.sqrt(np.nanmean((p - t) ** 2))
        mae     = np.nanmean(np.abs(p - t))
        dir_acc = np.nanmean((p > 0) == (t > 0)) * 100

        # Convert log-returns to % for display
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
    if not os.path.exists(MODEL_PATH):
        print(f"No trained model at {MODEL_PATH} — run train.py first.")
    elif not os.path.exists(f"{CACHE_DIR}/test_X.npy"):
        print(f"No cache at {CACHE_DIR} — run train.py first.")
    elif not os.path.exists(f"{CACHE_DIR}/test_sym_ids.npy"):
        print(f"Cache is outdated — delete data/cache/ and rerun train.py.")
    else:
        run_validation()
