"""
build_nifty50_parquet.py
Builds data/nifty50_features.parquet with all Nifty 50 stocks.

Runs the same pipeline as notebooks/preprocessing.ipynb PLUS the
augment_features.py additions (vix_chg, 52w ratios, sector RS).
combined_features.parquet (10 stocks) is NOT touched.

Usage:
    python src/build_nifty50_parquet.py

Output:
    data/nifty50_features.parquet  (~120-180k rows, all 50 stocks)
"""
import os
import sys
import numpy as np
import pandas as pd
import duckdb

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(_ROOT, "data", "india_market (2).duckdb")
OUT_PATH = os.path.join(_ROOT, "data", "nifty50_features.parquet")

# Sector → best available NSE index benchmark
SECTOR_INDEX_MAP = {
    "Banks":                          "BANKNIFTY",
    "Fast Moving Consumer Goods":     "NIFTYFMCG",
    "Information Technology":         "NIFTYIT",
    "Oil, Gas & Consumable Fuels":    "NIFTYENERGY",
    "Capital Goods":                  "NIFTYINFRA",
    "Finance":                        "NIFTY50",
    "Insurance":                      "NIFTY50",
    "Telecommunication":              "NIFTY50",
    "Automobile and Auto Components": "NIFTY50",
    "Healthcare":                     "NIFTY50",
    "Metals & Mining":                "NIFTY50",
    "Realty":                         "NIFTY50",
    "Consumer Durables":              "NIFTY50",
    "Chemicals":                      "NIFTY50",
    "Power":                          "NIFTY50",
    "Cement & Cement Products":       "NIFTY50",
}

QUARTER_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}


# ── Technical indicators ───────────────────────────────────────────────────

def _rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).clip(0, 100)


def _macd(close, fast=12, slow=26, signal=9):
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd       = ema_fast - ema_slow
    macd_sig   = macd.ewm(span=signal,  adjust=False).mean()
    macd_hist  = macd - macd_sig
    return macd, macd_sig, macd_hist


def _bb_width(close, period=20):
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return ((upper - lower) / sma.replace(0, np.nan)).clip(0, 5)


def _log_ret(series, n):
    return np.log((series / series.shift(n)).clip(1e-9)).clip(-2, 2)


# ── Price adjustment (identical to notebook) ───────────────────────────────

def _parse_ratio(val, label=""):
    try:
        if isinstance(val, str) and ":" in val:
            a, b = val.split(":", 1)
            return float(a) / float(b)
        return float(val)
    except Exception:
        print(f"  WARNING: cannot parse ratio '{val}' ({label}) — skipping")
        return None


def adjust_prices(prices_df, corp_df):
    prices_df = prices_df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        prices_df[f"adj_{col}"] = prices_df[col].astype(float)

    for symbol in prices_df["symbol"].unique():
        sym_mask = prices_df["symbol"] == symbol
        sym_corp = corp_df[corp_df["symbol"] == symbol].sort_values("date", ascending=False)

        for _, row in sym_corp.iterrows():
            adate = row["date"]
            atype = str(row["action_type"]).upper()
            val   = row["value"]
            before = sym_mask & (prices_df["date"] < adate)
            if before.sum() == 0:
                continue

            factor = None
            if "SPLIT" in atype:
                r = _parse_ratio(val, f"{symbol} SPLIT")
                if r and r > 0:
                    factor = 1.0 / r
            elif "BONUS" in atype:
                r = _parse_ratio(val, f"{symbol} BONUS")
                if r and r > 0:
                    factor = 1.0 / (1.0 + r)
            elif "DIV" in atype:
                try:
                    d = float(val)
                    ex_rows = prices_df[sym_mask & (prices_df["date"] == adate)]
                    if len(ex_rows) > 0 and d > 0:
                        ex_cl = ex_rows["close"].iloc[0]
                        if ex_cl > d:
                            factor = (ex_cl - d) / ex_cl
                except Exception:
                    pass

            if factor is not None and 0 < factor < 10:
                for col in ["open", "high", "low", "close"]:
                    prices_df.loc[before, f"adj_{col}"] *= factor

    for col in ["open", "high", "low", "close"]:
        prices_df[col] = prices_df[f"adj_{col}"]
    return prices_df.drop(columns=[f"adj_{c}" for c in ["open", "high", "low", "close"]])


# ── Augmentation helpers ───────────────────────────────────────────────────

def add_vix_change_and_52w(df):
    if "india_vix" in df.columns:
        df["india_vix_chg"] = (
            df.groupby("symbol")["india_vix"]
            .pct_change(fill_method=None)
            .clip(-2, 2)
        )
    for sym, grp in df.groupby("symbol"):
        grp = grp.sort_values("date")
        if not {"high", "low", "close"}.issubset(df.columns):
            break
        high52 = grp["high"].rolling(252, min_periods=63).max()
        low52  = grp["low"].rolling(252, min_periods=63).min()
        df.loc[grp.index, "high52w_ratio"] = (
            (grp["close"] / high52.replace(0, np.nan) - 1).clip(-1, 0)
        )
        df.loc[grp.index, "low52w_ratio"] = (
            (grp["close"] / low52.replace(0, np.nan) - 1).clip(0, 5)
        )


def add_sector_rs(df, con, symbols):
    needed = list(set(
        SECTOR_INDEX_MAP.get(s, "NIFTY50")
        for s in df["sector"].unique()
        if isinstance(s, str)
    ) | {"NIFTY50"})

    sym_list = "'" + "', '".join(needed) + "'"
    idx_df = con.execute(f"""
        SELECT index_symbol, date, close
        FROM index_prices
        WHERE index_symbol IN ({sym_list})
        ORDER BY index_symbol, date
    """).df()
    if idx_df.empty:
        print("  WARNING: no index_prices data — skipping sector RS")
        return

    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_rets = {}
    for idx_sym, grp in idx_df.groupby("index_symbol"):
        c = grp.set_index("date")["close"].sort_index()
        idx_rets[idx_sym] = {"5d": _log_ret(c, 5), "20d": _log_ret(c, 20)}

    nif5  = idx_rets["NIFTY50"]["5d"]
    nif20 = idx_rets["NIFTY50"]["20d"]

    # Build a sector→index_sym map from sector column values
    # (df["sector"] at this point is still a string, not encoded)
    sector_col = "sector_name" if "sector_name" in df.columns else "sector"

    for sym, grp in df.groupby("symbol"):
        grp   = grp.sort_values("date")
        dates = pd.to_datetime(grp["date"])
        close = grp.set_index("date")["close"].sort_index()
        s5    = _log_ret(close, 5).reindex(dates.values)
        s20   = _log_ret(close, 20).reindex(dates.values)

        # Get the sector index for this stock
        sec_val   = grp[sector_col].iloc[0] if sector_col in grp.columns else "NIFTY50"
        sect_idx  = SECTOR_INDEX_MAP.get(str(sec_val), "NIFTY50")
        if sect_idx not in idx_rets:
            sect_idx = "NIFTY50"

        n5_d   = nif5.reindex(dates.values)
        n20_d  = nif20.reindex(dates.values)
        sc20_d = idx_rets[sect_idx]["20d"].reindex(dates.values)

        df.loc[grp.index, "rs_vs_nifty_5d"]   = (s5.values  - n5_d.values).clip(-1, 1)
        df.loc[grp.index, "rs_vs_nifty_20d"]  = (s20.values - n20_d.values).clip(-1, 1)
        df.loc[grp.index, "rs_vs_sector_20d"] = (s20.values - sc20_d.values).clip(-1, 1)


def add_fii_dii(df, con, symbols):
    sym_list = "'" + "', '".join(symbols) + "'"
    sh = con.execute(f"""
        SELECT symbol, quarter_end, fii_pct, dii_pct
        FROM shareholding
        WHERE symbol IN ({sym_list})
        ORDER BY symbol, quarter_end
    """).df()
    if sh.empty:
        df["fii_pct"] = np.nan
        df["dii_pct"] = np.nan
        df["fii_pct_chg"] = np.nan
        return
    sh["quarter_end"] = pd.to_datetime(sh["quarter_end"]).astype("datetime64[ns]")

    for sym, grp in df.groupby("symbol"):
        mask   = df["symbol"] == sym
        sym_df = df.loc[mask, ["date"]].sort_values("date").reset_index()
        sym_df["date"] = sym_df["date"].astype("datetime64[ns]")
        sh_sym = sh[sh["symbol"] == sym].sort_values("quarter_end").copy()
        sh_sym["fii_pct_chg"] = sh_sym["fii_pct"].diff()

        if sh_sym.empty:
            df.loc[mask, ["fii_pct", "dii_pct", "fii_pct_chg"]] = np.nan
            continue

        merged = pd.merge_asof(
            sym_df.sort_values("date"),
            sh_sym[["quarter_end", "fii_pct", "dii_pct", "fii_pct_chg"]],
            left_on="date", right_on="quarter_end", direction="backward",
        )
        idx = sym_df["index"].values
        df.loc[idx, "fii_pct"]     = merged["fii_pct"].values
        df.loc[idx, "dii_pct"]     = merged["dii_pct"].values
        df.loc[idx, "fii_pct_chg"] = merged["fii_pct_chg"].values


def add_eps_momentum(df, con, symbols):
    sym_list = "'" + "', '".join(symbols) + "'"
    fin = con.execute(f"""
        SELECT symbol, year, quarter, eps
        FROM financials
        WHERE symbol IN ({sym_list}) AND period = 'quarterly'
        ORDER BY symbol, year, quarter
    """).df()
    if fin.empty:
        df["eps_yoy"] = np.nan
        return

    fin["quarter_end"] = fin.apply(
        lambda r: (pd.Timestamp(year=int(r["year"]),
                                month=QUARTER_MONTH.get(int(r["quarter"]), 12),
                                day=1) + pd.offsets.MonthEnd(0)),
        axis=1,
    ).astype("datetime64[ns]")
    fin = fin.sort_values(["symbol", "year", "quarter"])
    fin["eps_lag4"] = fin.groupby(["symbol", "quarter"])["eps"].shift(1)
    fin["eps_yoy"]  = (
        (fin["eps"] / fin["eps_lag4"].replace(0, np.nan) - 1)
        .clip(-5, 5).fillna(0.0)
    )

    for sym, grp in df.groupby("symbol"):
        mask   = df["symbol"] == sym
        sym_df = df.loc[mask, ["date"]].sort_values("date").reset_index()
        sym_df["date"] = sym_df["date"].astype("datetime64[ns]")
        fin_sym = fin[fin["symbol"] == sym].sort_values("quarter_end")
        if fin_sym.empty:
            df.loc[mask, "eps_yoy"] = np.nan
            continue
        merged = pd.merge_asof(
            sym_df.sort_values("date"),
            fin_sym[["quarter_end", "eps_yoy"]],
            left_on="date", right_on="quarter_end", direction="backward",
        )
        df.loc[sym_df["index"].values, "eps_yoy"] = merged["eps_yoy"].values


# ── Main pipeline ──────────────────────────────────────────────────────────

def main():
    if not os.path.exists(DB_PATH):
        print(f"DuckDB not found at {DB_PATH}")
        sys.exit(1)

    print(f"Connecting to {DB_PATH} ...")
    con = duckdb.connect(DB_PATH, read_only=True)

    # ── 1. Get Nifty 50 constituents ──────────────────────────────────────
    constituents = con.execute("""
        SELECT DISTINCT symbol
        FROM index_constituents
        WHERE index_symbol = 'NIFTY50'
        ORDER BY symbol
    """).df()
    symbols = constituents["symbol"].tolist()
    print(f"  Nifty 50 symbols in DB: {len(symbols)}")
    if not symbols:
        print("  ERROR: no constituents found for NIFTY50 in index_constituents")
        sys.exit(1)

    sym_list = "'" + "', '".join(symbols) + "'"

    # ── 2. Prices ─────────────────────────────────────────────────────────
    print("\nFetching prices ...")
    prices_df = con.execute(f"""
        SELECT symbol, date, open, high, low, close, volume
        FROM prices
        WHERE symbol IN ({sym_list})
        ORDER BY symbol, date
    """).df()
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    print(f"  Raw prices: {prices_df.shape}")

    # Keep only symbols that actually have price data
    symbols = prices_df["symbol"].unique().tolist()
    sym_list = "'" + "', '".join(symbols) + "'"
    print(f"  Symbols with prices: {len(symbols)}")

    # ── 3. Corporate actions + price adjustment ───────────────────────────
    print("Fetching corporate actions and adjusting prices ...")
    corp_df = con.execute(f"""
        SELECT symbol, date, action_type, value, remarks
        FROM corporate_actions
        WHERE symbol IN ({sym_list})
        ORDER BY symbol, date
    """).df()
    corp_df["date"] = pd.to_datetime(corp_df["date"])
    print(f"  Corporate actions: {len(corp_df)}")

    prices_df = adjust_prices(prices_df, corp_df)
    print("  Prices adjusted.")

    # ── 4. Company metadata ───────────────────────────────────────────────
    print("Fetching company metadata ...")
    companies_df = con.execute(f"""
        SELECT symbol, company_name, sector, industry, cap_category, avg_mcap_cr
        FROM companies
        WHERE symbol IN ({sym_list})
    """).df()
    combined_df = pd.merge(prices_df, companies_df, on="symbol", how="left")

    # ── 5. Financials ─────────────────────────────────────────────────────
    print("Fetching financials ...")
    fin_df = con.execute(f"""
        SELECT symbol, year, quarter, period,
               revenue, net_profit, ebitda, eps, assets, liabilities,
               equity, debt, operating_cash_flow, free_cash_flow,
               book_value_per_share, operating_profit, ebit,
               shares_outstanding, cash_equivalents
        FROM financials
        WHERE symbol IN ({sym_list})
        ORDER BY symbol, year, quarter
    """).df()

    def _fin_date(row):
        try:
            y, p = int(row["year"]), row["period"]
            if p == "annual": return pd.Timestamp(f"{y+1}-03-31")
            if p == "Q1":     return pd.Timestamp(f"{y}-06-30")
            if p == "Q2":     return pd.Timestamp(f"{y}-09-30")
            if p == "Q3":     return pd.Timestamp(f"{y}-12-31")
            if p == "Q4":     return pd.Timestamp(f"{y+1}-03-31")
        except Exception:
            return pd.NaT
        return pd.NaT

    fin_df["date"] = fin_df.apply(_fin_date, axis=1)
    fin_df = fin_df.drop(columns=["year", "quarter", "period"]).dropna(subset=["date"])
    fin_df = fin_df.drop_duplicates(subset=["symbol", "date"])

    # ── 6. Shareholding ───────────────────────────────────────────────────
    print("Fetching shareholding ...")
    sh_df = con.execute(f"""
        SELECT symbol, quarter_end, promoter_pct, fii_pct, dii_pct, public_pct
        FROM shareholding
        WHERE symbol IN ({sym_list})
        ORDER BY symbol, quarter_end
    """).df()
    sh_df["quarter_end"] = pd.to_datetime(sh_df["quarter_end"])
    sh_df_m = sh_df.rename(columns={"quarter_end": "date"})

    # ── 7. Company betas ──────────────────────────────────────────────────
    betas_df = con.execute(f"""
        SELECT symbol, blended_beta FROM company_betas
        WHERE symbol IN ({sym_list})
    """).df()

    # ── 8. Macro data ─────────────────────────────────────────────────────
    print("Fetching macro data ...")
    macro_df = con.execute("SELECT date, metric, value FROM macro_data ORDER BY date").df()
    macro_df["date"] = pd.to_datetime(macro_df["date"]).astype("datetime64[ns]")
    macro_pivot = (
        macro_df.pivot_table(index="date", columns="metric", values="value")
        .reset_index()
    )
    macro_pivot.columns.name = None

    # ── 9. merge_asof joins ───────────────────────────────────────────────
    print("Merging all data sources ...")
    for df_ in [combined_df, fin_df, macro_pivot]:
        df_["date"] = df_["date"].astype("datetime64[ns]")
    sh_df_m["date"] = sh_df_m["date"].astype("datetime64[ns]")

    combined_df = combined_df.drop_duplicates(subset=["symbol", "date"])
    fin_df      = fin_df.drop_duplicates(subset=["symbol", "date"])
    combined_df = combined_df.sort_values("date").reset_index(drop=True)
    fin_df      = fin_df.sort_values("date").reset_index(drop=True)
    sh_df_m     = sh_df_m.sort_values("date").reset_index(drop=True)
    macro_pivot = macro_pivot.sort_values("date").reset_index(drop=True)

    combined_df = pd.merge_asof(combined_df, fin_df,    on="date", by="symbol", direction="backward")
    combined_df = pd.merge_asof(combined_df, sh_df_m,   on="date", by="symbol", direction="backward")
    combined_df = pd.merge_asof(combined_df, macro_pivot, on="date", direction="backward")
    combined_df = pd.merge(combined_df, betas_df, on="symbol", how="left")

    # ── 10. Financial ratios ──────────────────────────────────────────────
    fin_fill = ["revenue", "net_profit", "equity", "debt", "ebitda", "operating_cash_flow", "book_value_per_share"]
    combined_df = combined_df.sort_values(["symbol", "date"])
    for col in [c for c in fin_fill if c in combined_df.columns]:
        combined_df[col] = combined_df.groupby("symbol")[col].ffill()

    combined_df["profit_margin"]   = combined_df["net_profit"] / combined_df["revenue"]
    combined_df["roe"]             = combined_df["net_profit"] / combined_df["equity"]
    combined_df["debt_to_equity"]  = combined_df["debt"]       / combined_df["equity"]
    combined_df["ebitda_margin"]   = combined_df["ebitda"]     / combined_df["revenue"]
    combined_df["book_to_price"]   = combined_df.get("book_value_per_share", np.nan) / combined_df["close"]
    combined_df["cash_flow_margin"] = combined_df.get("operating_cash_flow", np.nan) / combined_df["revenue"]
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ── 11. Technical indicators ──────────────────────────────────────────
    print("Computing technical indicators ...")
    combined_df["volume"] = combined_df["volume"].astype(float)
    combined_df = combined_df.sort_values(["symbol", "date"])

    tech_results = []
    for sym, grp in combined_df.groupby("symbol"):
        grp = grp.copy()
        c   = grp["close"]
        v   = grp["volume"]

        grp["daily_return"]    = c.pct_change()
        grp["5d_return"]       = c.pct_change(5)
        grp["20d_return"]      = c.pct_change(20)
        grp["50d_ma"]          = c.rolling(50).mean()
        grp["200d_ma"]         = c.rolling(200).mean()
        grp["20d_volatility"]  = grp["daily_return"].rolling(20).std()
        grp["20d_avg_volume"]  = v.rolling(20).mean()
        grp["volume_ratio"]    = v / grp["20d_avg_volume"]

        grp["rsi_14"] = _rsi(c)
        grp["macd"], grp["macd_signal"], grp["macd_hist"] = _macd(c)
        grp["bb_width"] = _bb_width(c)

        tech_results.append(grp)

    combined_df = pd.concat(tech_results).sort_values(["symbol", "date"]).reset_index(drop=True)

    # ── 12. Corporate action flags ────────────────────────────────────────
    div_df = corp_df[corp_df["action_type"].str.upper().str.contains("DIV")][
        ["symbol", "date", "value"]].copy()
    div_df["is_ex_dividend"]  = 1
    div_df["dividend_amount"] = pd.to_numeric(div_df["value"], errors="coerce").fillna(0.0)
    div_df = div_df.drop(columns=["value"])

    split_df = corp_df[corp_df["action_type"].str.upper().str.contains("SPLIT|BONUS")][
        ["symbol", "date", "value"]].copy()
    split_df["is_split"]      = 1
    split_df["split_factor"]  = split_df["value"].apply(lambda v: _parse_ratio(v) or 0.0)
    split_df = split_df.drop(columns=["value"])

    combined_df = combined_df.merge(
        div_df[["symbol", "date", "is_ex_dividend", "dividend_amount"]],
        on=["symbol", "date"], how="left")
    combined_df = combined_df.merge(
        split_df[["symbol", "date", "is_split", "split_factor"]],
        on=["symbol", "date"], how="left")

    combined_df["is_ex_dividend"] = combined_df["is_ex_dividend"].fillna(0).astype(int)
    combined_df["dividend_amount"] = combined_df["dividend_amount"].fillna(0.0)
    combined_df["is_split"]        = combined_df["is_split"].fillna(0).astype(int)
    combined_df["split_factor"]    = combined_df["split_factor"].fillna(0.0)

    # ── 13. NaN handling ─────────────────────────────────────────────────
    combined_df = combined_df[combined_df["200d_ma"].notna()].reset_index(drop=True)

    macro_cols  = [c for c in combined_df.columns if c.startswith("us_") or c in ["usd_inr", "wti_crude_usd"]]
    ratio_cols  = ["profit_margin", "roe", "debt_to_equity", "ebitda_margin", "book_to_price", "cash_flow_margin"]
    fill_cols   = macro_cols + ratio_cols
    combined_df[fill_cols] = combined_df.groupby("symbol")[fill_cols].transform("ffill")

    critical = ["daily_return", "5d_return", "20d_return", "50d_ma", "200d_ma", "20d_volatility", "volume_ratio"]
    combined_df = combined_df.dropna(subset=critical).reset_index(drop=True)
    print(f"  After NaN handling: {combined_df.shape}")

    # ── 14. Augmented features (vix_chg, 52w, sector RS, FII/DII, EPS) ──
    print("\nAdding augmented features ...")

    # Store sector_name before any potential encoding for sector RS lookup
    combined_df["sector_name"] = combined_df["sector"].astype(str)

    add_vix_change_and_52w(combined_df)
    add_sector_rs(combined_df, con, symbols)
    add_fii_dii(combined_df, con, symbols)
    add_eps_momentum(combined_df, con, symbols)

    combined_df = combined_df.drop(columns=["sector_name"], errors="ignore")

    # ── 15. Coverage report ────────────────────────────────────────────────
    aug_cols = ["india_vix_chg", "high52w_ratio", "low52w_ratio",
                "rs_vs_nifty_5d", "rs_vs_nifty_20d", "rs_vs_sector_20d",
                "fii_pct", "dii_pct", "fii_pct_chg", "eps_yoy"]
    print(f"\n{'='*55}")
    print(f"  Augmented column coverage:")
    for col in aug_cols:
        if col in combined_df.columns:
            pct = combined_df[col].notna().mean() * 100
            print(f"  {col:25s}: {pct:5.1f}% non-null")
    print(f"{'='*55}")

    # ── 16. Save ──────────────────────────────────────────────────────────
    con.close()

    print(f"\nSaving → {OUT_PATH}")
    combined_df.to_parquet(OUT_PATH, index=False)
    print(f"  Done. Shape: {combined_df.shape}")
    print(f"  Symbols: {sorted(combined_df['symbol'].unique().tolist())}")
    print(f"\nNext steps:")
    print(f"  1. Update train.py DATA_PATH to '{OUT_PATH}' to use Nifty 50 data")
    print(f"  2. Delete cache:  Remove-Item -Recurse -Force data\\cache")
    print(f"  3. Retrain:       python src/train.py")


if __name__ == "__main__":
    main()
