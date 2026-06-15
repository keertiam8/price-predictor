"""
augment_features.py — adds signal-rich features to combined_features.parquet from DuckDB.

New columns added:
  rs_vs_nifty_5d     : stock 5d log-return minus Nifty50 5d log-return
  rs_vs_nifty_20d    : stock 20d log-return minus Nifty50 20d log-return
  rs_vs_sector_20d   : stock 20d log-return minus its sector benchmark 20d log-return
  fii_pct            : FII % holdings (quarterly, forward-filled to daily)
  dii_pct            : DII % holdings (quarterly, forward-filled to daily)
  fii_pct_chg        : quarter-on-quarter change in FII % (signal of institutional direction)
  eps_yoy            : year-on-year quarterly EPS growth rate (fundamental momentum)

Also adds in-place (computed from existing parquet columns, no DuckDB needed):
  india_vix_chg      : daily % change in India VIX (fear momentum)
  high52w_ratio      : (close / 52-week-high) - 1  [<= 0: how far from highs]
  low52w_ratio       : (close / 52-week-low) - 1   [>= 0: how far above lows]

Run once, then delete data/cache/ before retraining:
  python src/augment_features.py
  Remove-Item -Recurse -Force data\\cache   # PowerShell
"""
import os
import sys
import duckdb
import numpy as np
import pandas as pd

DB_PATH      = "data/india_market (2).duckdb"
PARQUET_PATH = "data/combined_features.parquet"

SYMBOLS = [
    "BAJFINANCE", "BHARTIARTL", "HDFCBANK", "HINDUNILVR",
    "ICICIBANK",  "LICI",       "LT",       "RELIANCE",
    "SBIN",       "TCS",
]

# Best available sector benchmark for each stock
SECTOR_INDEX_MAP = {
    "BAJFINANCE":  "NIFTY50",      # NBFC — no dedicated NSE sector index
    "BHARTIARTL":  "NIFTY50",      # Telecom — no dedicated NSE sector index
    "HDFCBANK":    "BANKNIFTY",
    "HINDUNILVR":  "NIFTYFMCG",
    "ICICIBANK":   "BANKNIFTY",
    "LICI":        "NIFTY50",      # Insurance — no dedicated NSE sector index
    "LT":          "NIFTYINFRA",
    "RELIANCE":    "NIFTYENERGY",
    "SBIN":        "PSUBANK",
    "TCS":         "NIFTYIT",
}

QUARTER_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}


# ── helpers ────────────────────────────────────────────────────────────────────

def _log_ret(series, n):
    """n-day log return, clipped at sensible bounds."""
    return np.log((series / series.shift(n)).clip(1e-9)).clip(-2, 2)


def _sym_str(syms):
    return "'" + "', '".join(syms) + "'"


# ── feature builders ───────────────────────────────────────────────────────────

def add_vix_change_and_52w(df):
    """Computed purely from existing parquet columns — no DuckDB needed."""
    print("  Computing india_vix_chg and 52-week high/low proximity...")

    if "india_vix" in df.columns:
        df["india_vix_chg"] = (
            df.groupby("symbol")["india_vix"]
            .pct_change(fill_method=None)
            .clip(-2, 2)
        )

    for sym in SYMBOLS:
        mask = df["symbol"] == sym
        g    = df.loc[mask].sort_values("date")

        if "high" in df.columns and "low" in df.columns and "close" in df.columns:
            high52 = g["high"].rolling(252, min_periods=63).max()
            low52  = g["low"].rolling(252, min_periods=63).min()
            df.loc[g.index, "high52w_ratio"] = (
                (g["close"] / high52.replace(0, np.nan) - 1).clip(-1, 0)
            )
            df.loc[g.index, "low52w_ratio"] = (
                (g["close"] / low52.replace(0, np.nan) - 1).clip(0, 5)
            )

    print("    Added: india_vix_chg, high52w_ratio, low52w_ratio")


def add_sector_rs(df, con):
    """Relative strength of stock vs Nifty50 and vs sector benchmark."""
    print("  Fetching sector index prices...")
    needed = list(set(SECTOR_INDEX_MAP.values()))
    idx_df = con.execute(f"""
        SELECT index_symbol, date, close
        FROM index_prices
        WHERE index_symbol IN ({_sym_str(needed)})
        ORDER BY index_symbol, date
    """).df()
    idx_df["date"] = pd.to_datetime(idx_df["date"])

    # Pre-compute 5d and 20d log-returns per index
    idx_rets = {}
    for idx_sym, grp in idx_df.groupby("index_symbol"):
        close = grp.set_index("date")["close"].sort_index()
        idx_rets[idx_sym] = {
            "5d":  _log_ret(close, 5),
            "20d": _log_ret(close, 20),
        }

    nif5  = idx_rets["NIFTY50"]["5d"]
    nif20 = idx_rets["NIFTY50"]["20d"]

    for sym in SYMBOLS:
        mask = df["symbol"] == sym
        g    = df.loc[mask].sort_values("date")
        dates = pd.to_datetime(g["date"])

        close  = g.set_index("date")["close"].sort_index()
        s5     = _log_ret(close, 5).reindex(dates.values)
        s20    = _log_ret(close, 20).reindex(dates.values)

        sect_idx = SECTOR_INDEX_MAP[sym]
        n5_d   = nif5.reindex(dates.values)
        n20_d  = nif20.reindex(dates.values)
        sc20_d = idx_rets[sect_idx]["20d"].reindex(dates.values)

        df.loc[g.index, "rs_vs_nifty_5d"]   = (s5.values  - n5_d.values).clip(-1, 1)
        df.loc[g.index, "rs_vs_nifty_20d"]  = (s20.values - n20_d.values).clip(-1, 1)
        df.loc[g.index, "rs_vs_sector_20d"] = (s20.values - sc20_d.values).clip(-1, 1)

    print("    Added: rs_vs_nifty_5d, rs_vs_nifty_20d, rs_vs_sector_20d")


def add_fii_dii(df, con):
    """Quarterly FII/DII % holdings forward-filled to daily."""
    print("  Fetching FII/DII shareholding...")
    sh = con.execute(f"""
        SELECT symbol, quarter_end, fii_pct, dii_pct
        FROM shareholding
        WHERE symbol IN ({_sym_str(SYMBOLS)})
        ORDER BY symbol, quarter_end
    """).df()
    sh["quarter_end"] = pd.to_datetime(sh["quarter_end"]).astype("datetime64[ns]")

    found = sh["symbol"].unique().tolist()
    print(f"    Shareholding data found for: {found}")

    for sym in SYMBOLS:
        mask   = df["symbol"] == sym
        sym_df = df.loc[mask, ["date"]].sort_values("date").reset_index()
        sym_df["date"] = sym_df["date"].astype("datetime64[ns]")

        sh_sym = sh[sh["symbol"] == sym].sort_values("quarter_end").copy()
        sh_sym["fii_pct_chg"] = sh_sym["fii_pct"].diff()

        if len(sh_sym) == 0:
            df.loc[mask, "fii_pct"]     = np.nan
            df.loc[mask, "dii_pct"]     = np.nan
            df.loc[mask, "fii_pct_chg"] = np.nan
            continue

        # Forward-fill quarterly snapshot to every trading day
        merged = pd.merge_asof(
            sym_df.sort_values("date"),
            sh_sym[["quarter_end", "fii_pct", "dii_pct", "fii_pct_chg"]],
            left_on="date", right_on="quarter_end",
            direction="backward",
        )
        orig_idx = sym_df["index"].values
        df.loc[orig_idx, "fii_pct"]     = merged["fii_pct"].values
        df.loc[orig_idx, "dii_pct"]     = merged["dii_pct"].values
        df.loc[orig_idx, "fii_pct_chg"] = merged["fii_pct_chg"].values

    print("    Added: fii_pct, dii_pct, fii_pct_chg")


def add_eps_momentum(df, con):
    """Year-on-year quarterly EPS growth, forward-filled to daily."""
    print("  Fetching quarterly EPS from financials...")
    fin = con.execute(f"""
        SELECT symbol, year, quarter, eps
        FROM financials
        WHERE symbol IN ({_sym_str(SYMBOLS)}) AND period = 'quarterly'
        ORDER BY symbol, year, quarter
    """).df()

    found = fin["symbol"].unique().tolist()
    print(f"    EPS data found for: {found}")

    if len(fin) == 0:
        df["eps_yoy"] = np.nan
        return

    # Map (year, quarter) to approximate quarter-end date
    fin["quarter_end"] = fin.apply(
        lambda r: (pd.Timestamp(year=int(r["year"]),
                                month=QUARTER_MONTH.get(int(r["quarter"]), 12),
                                day=1) + pd.offsets.MonthEnd(0)),
        axis=1,
    ).astype("datetime64[ns]")
    fin = fin.sort_values(["symbol", "year", "quarter"])

    # YoY EPS: compare same quarter to previous year
    fin["eps_lag4"] = fin.groupby(["symbol", "quarter"])["eps"].shift(1)
    fin["eps_yoy"]  = (
        (fin["eps"] / fin["eps_lag4"].replace(0, np.nan) - 1)
        .clip(-5, 5)
        .fillna(0.0)
    )

    for sym in SYMBOLS:
        mask   = df["symbol"] == sym
        sym_df = df.loc[mask, ["date"]].sort_values("date").reset_index()
        sym_df["date"] = sym_df["date"].astype("datetime64[ns]")

        fin_sym = fin[fin["symbol"] == sym].sort_values("quarter_end")
        if len(fin_sym) == 0:
            df.loc[mask, "eps_yoy"] = np.nan
            continue

        merged = pd.merge_asof(
            sym_df.sort_values("date"),
            fin_sym[["quarter_end", "eps_yoy"]],
            left_on="date", right_on="quarter_end",
            direction="backward",
        )
        orig_idx = sym_df["index"].values
        df.loc[orig_idx, "eps_yoy"] = merged["eps_yoy"].values

    print("    Added: eps_yoy")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading parquet: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  {len(df):,} rows × {len(df.columns)} columns\n")

    # Drop any columns from a previous augment run to recompute fresh
    new_feature_names = [
        "india_vix_chg", "high52w_ratio", "low52w_ratio",
        "rs_vs_nifty_5d", "rs_vs_nifty_20d", "rs_vs_sector_20d",
        "fii_pct", "dii_pct", "fii_pct_chg", "eps_yoy",
    ]
    df = df.drop(columns=[c for c in new_feature_names if c in df.columns])

    # ── No-DuckDB features (computed from existing columns) ───────────────
    add_vix_change_and_52w(df)

    # ── DuckDB features ───────────────────────────────────────────────────
    if not os.path.exists(DB_PATH):
        print(f"\nDuckDB not found at: {DB_PATH}")
        print("  Skipping sector RS, FII/DII, EPS features.")
    else:
        print(f"\nConnecting to DuckDB: {DB_PATH}")
        con = duckdb.connect(DB_PATH, read_only=True)
        add_sector_rs(df, con)
        add_fii_dii(df, con)
        add_eps_momentum(df, con)
        con.close()

    # ── Coverage report ───────────────────────────────────────────────────
    added = [c for c in new_feature_names if c in df.columns]
    print(f"\n{'='*60}")
    print(f"  New columns added: {len(added)}")
    print(f"{'='*60}")
    for col in added:
        pct = df[col].notna().mean() * 100
        print(f"  {col:25s}: {pct:5.1f}% non-null")

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\nSaving updated parquet → {PARQUET_PATH} ...")
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"  Done. Shape: {df.shape}")

    print(f"\n{'='*60}")
    print("  NEXT STEPS:")
    print("  1. Delete the cache so train.py rebuilds features:")
    print("       Remove-Item -Recurse -Force data\\cache")
    print("  2. Retrain:")
    print("       python src/train.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
