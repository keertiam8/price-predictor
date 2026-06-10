import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Model definition ──────────────────────────────────────────────────────
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


# ── Config ────────────────────────────────────────────────────────────────
CACHE_DIR   = "data/cache"
MODEL_PATH  = "models/best_lstm_attention.pt"
HORIZONS    = [5, 10, 20]
BATCH_SIZE  = 128
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(preds, actuals):
    rmse    = np.sqrt(np.mean((preds - actuals) ** 2))
    mae     = np.mean(np.abs(preds - actuals))
    mape    = np.mean(np.abs((preds - actuals) / (actuals + 1e-8))) * 100
    dir_acc = np.mean(np.sign(preds - actuals) == np.sign(actuals)) * 100
    return rmse, mae, mape, dir_acc


def print_horizon_block(horizon, preds, actuals):
    rmse, mae, mape, dir_acc = compute_metrics(preds, actuals)

    print(f"\n{'='*52}")
    print(f"  {horizon}-DAY HORIZON")
    print(f"{'='*52}")
    print(f"  RMSE              : {rmse:>10.2f}")
    print(f"  MAE               : {mae:>10.2f}")
    print(f"  MAPE              : {mape:>9.2f}%")
    print(f"  Directional Acc   : {dir_acc:>9.2f}%")
    print(f"{'─'*52}")
    print(f"  Sample predictions (first 8 sequences):")
    print(f"  {'#':>4}  {'Actual':>10}  {'Predicted':>10}  {'Error':>8}  {'Dir':>5}")
    print(f"  {'─'*46}")
    for j in range(min(8, len(preds))):
        actual = actuals[j]
        pred   = preds[j]
        err    = pred - actual
        direction = "UP" if pred > actual else "DOWN"
        correct   = direction == ("UP" if actual > 0 else "DOWN")
        tick      = "+" if correct else "-"
        print(f"  {j+1:>4}  {actual:>10.2f}  {pred:>10.2f}  {err:>+8.2f}  {tick} {direction}")


def print_combined_block(all_preds, all_actuals):
    print(f"\n{'='*52}")
    print(f"  COMBINED (all horizons)")
    print(f"{'='*52}")
    print(f"  {'Horizon':>10}  {'RMSE':>8}  {'MAE':>8}  {'MAPE':>8}  {'Dir Acc':>9}")
    print(f"  {'─'*50}")

    all_rmse, all_mae, all_mape, all_dir = [], [], [], []
    for i, h in enumerate(HORIZONS):
        rmse, mae, mape, dir_acc = compute_metrics(all_preds[:, i], all_actuals[:, i])
        all_rmse.append(rmse)
        all_mae.append(mae)
        all_mape.append(mape)
        all_dir.append(dir_acc)
        print(f"  {h:>8d}d  {rmse:>8.2f}  {mae:>8.2f}  {mape:>7.2f}%  {dir_acc:>8.2f}%")

    print(f"  {'─'*50}")
    print(f"  {'Average':>10}  {np.mean(all_rmse):>8.2f}  {np.mean(all_mae):>8.2f}"
          f"  {np.mean(all_mape):>7.2f}%  {np.mean(all_dir):>8.2f}%")
    print(f"{'='*52}")


def run_validation():
    print(f"Loading model from {MODEL_PATH} ...")
    checkpoint         = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    tgt_scaler         = checkpoint["tgt_scaler"]
    input_size         = checkpoint["feature_cols_count"]

    model = LSTMAttentionModel(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, len(HORIZONS)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"  Device: {DEVICE}")

    print(f"\nLoading test data from {CACHE_DIR} ...")
    test_X = np.load(f"{CACHE_DIR}/test_X.npy")
    test_y = np.load(f"{CACHE_DIR}/test_y.npy")
    print(f"  Test sequences: {len(test_X):,}")

    loader = DataLoader(
        TensorDataset(torch.tensor(test_X, dtype=torch.float32),
                      torch.tensor(test_y, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    all_preds, all_tgts = [], []
    with torch.no_grad():
        for X, y in loader:
            pred, _ = model(X.to(DEVICE))
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(y.numpy())

    preds_price = tgt_scaler.inverse_transform(np.concatenate(all_preds))
    tgts_price  = tgt_scaler.inverse_transform(np.concatenate(all_tgts))

    # ── Per-horizon blocks ────────────────────────────────────────────────
    for i, h in enumerate(HORIZONS):
        print_horizon_block(h, preds_price[:, i], tgts_price[:, i])

    # ── Combined summary ──────────────────────────────────────────────────
    print_combined_block(preds_price, tgts_price)


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"No trained model at {MODEL_PATH} — run train.py first.")
    elif not os.path.exists(f"{CACHE_DIR}/test_X.npy"):
        print(f"No cache at {CACHE_DIR} — run train.py first.")
    else:
        run_validation()
