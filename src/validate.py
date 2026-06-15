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


class TanhGateLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight_ih = nn.Linear(input_size,  4 * hidden_size, bias=True)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x, state):
        h, c = state
        gates = self.weight_ih(x) + self.weight_hh(h)
        i, f, g, o = gates.chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.tanh(o) * torch.tanh(c_new)   # tanh output gate
        return h_new, c_new


class TanhGateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        sizes = [input_size] + [hidden_size] * num_layers
        self.cells = nn.ModuleList(TanhGateLSTMCell(sizes[i], hidden_size) for i in range(num_layers))
        self.drop  = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None

    def forward(self, x):
        B, T, _ = x.shape
        dev = x.device
        h = [torch.zeros(B, self.hidden_size, device=dev) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_size, device=dev) for _ in range(self.num_layers)]
        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for layer, cell in enumerate(self.cells):
                h[layer], c[layer] = cell(inp, (h[layer], c[layer]))
                inp = h[layer]
                if self.drop and layer < self.num_layers - 1:
                    inp = self.drop(inp)
            outputs.append(h[-1])
        return torch.stack(outputs, dim=1), (torch.stack(h, dim=0), torch.stack(c, dim=0))


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        e = self.score(lstm_out).squeeze(-1)
        a = torch.softmax(e, dim=1).unsqueeze(-1)
        return (a * lstm_out).sum(dim=1), a.squeeze(-1)


class LSTMAttentionModel(nn.Module):
    """Dual-head LSTM+attention (must match train.py).
    Uses TanhGateLSTM (tanh output gate) — must stay in sync with train.py."""
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_outputs):
        super().__init__()
        self.lstm      = TanhGateLSTM(input_size, hidden_size, num_layers,
                                      dropout=dropout if num_layers > 1 else 0.0)
        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.reg_head  = nn.Linear(hidden_size, num_outputs)
        self.cls_head  = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        lstm_out, _      = self.lstm(x)
        context, weights = self.attention(lstm_out)
        z = self.dropout(context)
        return self.reg_head(z), self.cls_head(z), weights


def run_validation():
    print(f"Loading model from {MODEL_PATH} ...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    input_size  = checkpoint["feature_cols_count"]
    hidden_size = checkpoint.get("hidden_size", HIDDEN_SIZE)
    num_layers  = checkpoint.get("num_layers",  NUM_LAYERS)
    dropout     = checkpoint.get("dropout",     DROPOUT)

    model = LSTMAttentionModel(input_size, hidden_size, num_layers, dropout, len(HORIZONS)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"  Device: {DEVICE}")

    target_scalers = checkpoint.get("target_scalers", {})

    print(f"\nLoading test data from {CACHE_DIR} ...")
    test_X = np.load(f"{CACHE_DIR}/test_X.npy")
    test_y = np.load(f"{CACHE_DIR}/test_y.npy")
    print(f"  Test sequences: {len(test_X):,}")

    # Run inference in batches
    all_reg, all_cls = [], []
    with torch.no_grad():
        for i in range(0, len(test_X), BATCH_SIZE):
            X_batch = torch.tensor(test_X[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            reg_pred, cls_logits, _ = model(X_batch)
            all_reg.append(reg_pred.cpu().numpy())
            all_cls.append(cls_logits.cpu().numpy())

    preds_std = np.concatenate(all_reg)   # regression head — standardised magnitude
    cls_logit = np.concatenate(all_cls)   # classification head — direction logits
    tgts_std  = test_y                     # standardised targets

    def invert(arr_std, h):
        """Invert StandardScaler to get raw log-returns."""
        if h in target_scalers:
            return target_scalers[h].inverse_transform(arr_std.reshape(-1, 1)).ravel()
        return arr_std  # fallback if no scaler saved

    def horizon_block(h_idx, h):
        p = invert(preds_std[:, h_idx], h)   # raw log-return (regression magnitude)
        t = invert(tgts_std[:, h_idx],  h)   # raw log-return
        pred_up = cls_logit[:, h_idx] > 0    # direction from classification head

        rmse    = np.sqrt(np.nanmean((p - t) ** 2))
        mae     = np.nanmean(np.abs(p - t))
        dir_acc = np.nanmean(pred_up == (t > 0)) * 100

        # Convert log-returns to % for display
        p_pct = (np.exp(p) - 1) * 100
        t_pct = (np.exp(t) - 1) * 100

        print(f"\n{'='*60}")
        print(f"  {h}-DAY HORIZON")
        print(f"{'='*60}")
        print(f"  RMSE (log-return)    : {rmse:>10.4f}")
        print(f"  MAE  (log-return)    : {mae:>10.4f}")
        print(f"  Directional Acc      : {dir_acc:>9.2f}%")
        print(f"  (Dir from classifier head; actual return > 0)")
        print(f"{'─'*60}")
        print(f"  Sample predictions (first 8 test sequences):")
        print(f"  {'#':>4}  {'Actual %':>10}  {'Pred %':>10}  {'Dir':>4}  {'Correct':>7}")
        print(f"  {'─'*52}")
        for j in range(min(8, len(p))):
            d_str   = "UP" if pred_up[j] else "DN"
            correct = pred_up[j] == (t[j] > 0)
            tick    = "YES" if correct else "NO"
            print(f"  {j+1:>4}  {t_pct[j]:>+9.2f}%  {p_pct[j]:>+9.2f}%  {d_str:>4}  {tick:>7}")

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
