"""
baseline_xgb.py — XGBoost directional baseline using the same cached features as the LSTM.

Usage:
    python src/baseline_xgb.py

Interpretation:
    If XGBoost Skill ≈ 0%  → features have no signal → need better features, not a better model.
    If XGBoost Skill > +3% → features DO have signal → LSTM architecture is the problem.

Features: last timestep of each 60-step sequence (most recent observation at prediction time).
"""
import os
import sys
import pickle
import numpy as np

CACHE_DIR = "data/cache"
HORIZONS  = [3, 7, 14]

try:
    import xgboost as xgb
    _BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _BACKEND = "sklearn-gbm"
    print("  xgboost not found — using sklearn GradientBoostingClassifier instead.")
    print("  Install with:  pip install xgboost\n")


def _load_cache():
    if not os.path.exists(f"{CACHE_DIR}/meta.pkl"):
        print(f"No cache at {CACHE_DIR} — run train.py first.")
        sys.exit(1)
    train_X = np.load(f"{CACHE_DIR}/train_X.npy")   # (N, 60, F)
    train_y = np.load(f"{CACHE_DIR}/train_y.npy")   # (N, H) standardized
    val_X   = np.load(f"{CACHE_DIR}/val_X.npy")
    val_y   = np.load(f"{CACHE_DIR}/val_y.npy")
    test_X  = np.load(f"{CACHE_DIR}/test_X.npy")
    test_y  = np.load(f"{CACHE_DIR}/test_y.npy")
    with open(f"{CACHE_DIR}/meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return train_X, train_y, val_X, val_y, test_X, test_y, meta


def _skill(y_true, y_pred):
    acc      = np.mean(y_true == y_pred) * 100
    baseline = y_true.mean() * 100
    return acc - baseline, acc, baseline


def _train_xgb(Xtr, ytr, Xva, yva):
    scale_pw = (1 - ytr.mean()) / max(ytr.mean(), 1e-6)
    clf = xgb.XGBClassifier(
        n_estimators        = 500,
        max_depth           = 4,
        learning_rate       = 0.03,
        subsample           = 0.8,
        colsample_bytree    = 0.7,
        min_child_weight    = 5,
        reg_alpha           = 0.1,
        reg_lambda          = 1.0,
        scale_pos_weight    = scale_pw,
        eval_metric         = "logloss",
        early_stopping_rounds = 30,
        verbosity           = 0,
        random_state        = 42,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return clf


def _train_sklearn(Xtr, ytr):
    clf = GradientBoostingClassifier(
        n_estimators  = 300,
        max_depth     = 3,
        learning_rate = 0.05,
        subsample     = 0.8,
        random_state  = 42,
    )
    clf.fit(Xtr, ytr)
    return clf


def _run(feature_label, Xtr_feat, Xva_feat, Xte_feat,
         train_y, val_y, test_y, zt):

    print(f"\n{'='*72}")
    print(f"  Feature set: {feature_label}")
    print(f"{'='*72}")
    print(f"  {'Horizon':>8}  {'Train Skill':>12}  {'Val Skill':>10}  {'Test Skill':>10}"
          f"  {'Test Acc':>9}  {'Baseline':>9}")
    print("  " + "-" * 62)

    for h_idx, h in enumerate(HORIZONS):
        ytr = (train_y[:, h_idx] > zt[h_idx]).astype(int)
        yva = (val_y[:, h_idx]   > zt[h_idx]).astype(int)
        yte = (test_y[:, h_idx]  > zt[h_idx]).astype(int)

        if _BACKEND == "xgboost":
            clf = _train_xgb(Xtr_feat, ytr, Xva_feat, yva)
        else:
            clf = _train_sklearn(Xtr_feat, ytr)

        tr_skill, tr_acc, tr_base = _skill(ytr, clf.predict(Xtr_feat))
        va_skill, va_acc, va_base = _skill(yva, clf.predict(Xva_feat))
        te_skill, te_acc, te_base = _skill(yte, clf.predict(Xte_feat))

        print(f"  {h:>6d}d  {tr_skill:>+11.2f}%  {va_skill:>+9.2f}%  {te_skill:>+9.2f}%"
              f"  {te_acc:>8.2f}%  {te_base:>8.2f}%")

        # Feature importance (top 5, only for last-step run)
        if "last step" in feature_label and hasattr(clf, "feature_importances_"):
            top = np.argsort(clf.feature_importances_)[-5:][::-1]
            feats = [f"f{i}" for i in range(Xtr_feat.shape[1])]
            try:
                with open(f"{CACHE_DIR}/meta.pkl", "rb") as f:
                    meta = pickle.load(f)
                feats = meta.get("feature_cols", feats)
            except Exception:
                pass
            top_names = [(feats[i] if i < len(feats) else f"f{i}",
                          clf.feature_importances_[i]) for i in top]
            print(f"          top features: " +
                  ", ".join(f"{n}({v:.3f})" for n, v in top_names))

    print("  " + "-" * 62)


def main():
    print(f"Loading cached data from {CACHE_DIR} ...")
    train_X, train_y, val_X, val_y, test_X, test_y, meta = _load_cache()
    target_scalers = meta["target_scalers"]

    zt = np.array([float(target_scalers[h].transform([[0.0]])[0, 0])
                   for h in HORIZONS])

    print(f"  Train: {len(train_X):,}  Val: {len(val_X):,}  Test: {len(test_X):,} sequences")
    print(f"  Window: {train_X.shape[1]} steps × {train_X.shape[2]} features")
    print(f"  Backend: {_BACKEND}")

    # Feature set 1: last timestep only (most recent observation)
    _run("last step (t=60)",
         train_X[:, -1, :], val_X[:, -1, :], test_X[:, -1, :],
         train_y, val_y, test_y, zt)

    # Feature set 2: mean across window (aggregated signal)
    _run("mean of window (t=1..60)",
         train_X.mean(axis=1), val_X.mean(axis=1), test_X.mean(axis=1),
         train_y, val_y, test_y, zt)

    # Feature set 3: last 5 timesteps flattened (short-term context)
    _run("last 5 steps flattened",
         train_X[:, -5:, :].reshape(len(train_X), -1),
         val_X[:,   -5:, :].reshape(len(val_X),   -1),
         test_X[:,  -5:, :].reshape(len(test_X),  -1),
         train_y, val_y, test_y, zt)

    print(f"\n{'='*72}")
    print("  INTERPRETATION")
    print(f"{'='*72}")
    print("  Val Skill ≈ 0%  → features have no directional signal for this horizon")
    print("  Val Skill > +3% → features have signal; LSTM architecture needs fixing")
    print("  Train Skill >> Val Skill → XGBoost is overfitting too; reduce depth/trees")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
