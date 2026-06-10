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
    test_X  = np.load(f"{CACHE_DIR}/test_X.npy")
    test_y  = np.load(f"{CACHE_DIR}/test_y.npy")
    test_cc = np.load(f"{CACHE_DIR}/test_cc.npy")   # current close (scaled, per-symbol)
    print(f"  Test sequences: {len(test_X):,}")

    # Run inference in batches
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_X), BATCH_SIZE):
            X_batch = torch.tensor(test_X[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            pred, _ = model(X_batch)
            all_preds.append(pred.cpu().numpy())

    preds_scaled = np.concatenate(all_preds)   # still in per-symbol scaled space
    tgts_scaled  = test_y

    # ── Inverse transform per symbol ────────────────────────────────────
    # Sequences in cache are ordered: all sym0 then all sym1 etc. (same order as training)
    sym_ids      = sorted(symbol_scalers.keys())
    n_total      = len(preds_scaled)
    n_per_sym    = n_total // len(sym_ids)

    preds_price  = np.zeros_like(preds_scaled)
    tgts_price   = np.zeros_like(tgts_scaled)

    for i, sym_id in enumerate(sym_ids):
        start = i * n_per_sym
        end   = (i + 1) * n_per_sym if i < len(sym_ids) - 1 else n_total
        sc    = symbol_scalers[sym_id]["tgt"]
        preds_price[start:end] = sc.inverse_transform(preds_scaled[start:end])
        tgts_price[start:end]  = sc.inverse_transform(tgts_scaled[start:end])

    # ── Correct direction formula ─────────────────────────────────────────
    # actual_dir = future_price > current_price
    # pred_dir   = predicted_price > current_price
    # (compared in scaled space — per-symbol scaling makes this valid)

    def horizon_block(h_idx, h):
        preds  = preds_price[:, h_idx]
        actuals = tgts_price[:, h_idx]
        cc      = test_cc

        rmse    = np.sqrt(np.mean((preds - actuals) ** 2))
        mae     = np.mean(np.abs(preds - actuals))
        mape    = np.mean(np.abs((preds - actuals) / (np.abs(actuals) + 1e-8))) * 100
        dir_acc = np.mean((preds_scaled[:, h_idx] > cc) ==
                          (tgts_scaled[:, h_idx]  > cc)) * 100

        print(f"\n{'='*52}")
        print(f"  {h}-DAY HORIZON")
        print(f"{'='*52}")
        print(f"  RMSE              : {rmse:>10.2f}")
        print(f"  MAE               : {mae:>10.2f}")
        print(f"  MAPE              : {mape:>9.2f}%")
        print(f"  Directional Acc   : {dir_acc:>9.2f}%")
        print(f"  (Dir Acc: predicted_price > current_close == actual_future > current_close)")
        print(f"{'─'*52}")
        print(f"  Sample predictions (first 8 sequences):")
        print(f"  {'#':>4}  {'Actual':>10}  {'Predicted':>10}  {'Error':>9}  {'Correct':>7}")
        print(f"  {'─'*48}")
        for j in range(min(8, len(preds))):
            err       = preds[j] - actuals[j]
            pred_up   = preds_scaled[j, h_idx] > cc[j]
            actual_up = tgts_scaled[j, h_idx]  > cc[j]
            correct   = pred_up == actual_up
            tick      = "YES" if correct else "NO"
            print(f"  {j+1:>4}  {actuals[j]:>10.2f}  {preds[j]:>10.2f}  {err:>+9.2f}  {tick:>7}")

        return rmse, mae, mape, dir_acc

    all_rmse, all_mae, all_mape, all_dir = [], [], [], []
    for i, h in enumerate(HORIZONS):
        rmse, mae, mape, dir_acc = horizon_block(i, h)
        all_rmse.append(rmse)
        all_mae.append(mae)
        all_mape.append(mape)
        all_dir.append(dir_acc)

    print(f"\n{'='*56}")
    print(f"  COMBINED SUMMARY (all horizons)")
    print(f"{'='*56}")
    print(f"  {'Horizon':>10}  {'RMSE':>8}  {'MAE':>8}  {'MAPE':>8}  {'Dir Acc':>9}")
    print(f"  {'─'*52}")
    for i, h in enumerate(HORIZONS):
        print(f"  {h:>8d}d  {all_rmse[i]:>8.2f}  {all_mae[i]:>8.2f}"
              f"  {all_mape[i]:>7.2f}%  {all_dir[i]:>8.2f}%")
    print(f"  {'─'*52}")
    print(f"  {'Average':>10}  {np.mean(all_rmse):>8.2f}  {np.mean(all_mae):>8.2f}"
          f"  {np.mean(all_mape):>7.2f}%  {np.mean(all_dir):>8.2f}%")
    print(f"{'='*56}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"No trained model at {MODEL_PATH} — run train.py first.")
    elif not os.path.exists(f"{CACHE_DIR}/test_X.npy"):
        print(f"No cache at {CACHE_DIR} — run train.py first.")
    else:
        run_validation()
