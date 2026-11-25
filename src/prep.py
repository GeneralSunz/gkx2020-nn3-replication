"""Build the modelling panel from the authors' datashare.csv.

GKX 2020 data conventions (from readme.txt of the authors' data + paper footnotes):
  * one row = (permno, month); DATE is a month-end trading date
  * 94 characteristics are already point-in-time shifted, so the *response* for a
    row dated month m is the return of month m itself
  * mom1m is the 1-month momentum, i.e. the return of the PREVIOUS month.
    => ret(month m) = mom1m(row of month m+1)  == mom1m.shift(-1) within permno
  * missing characteristics -> cross-sectional median of that month
  * characteristics -> cross-sectional rank scaled into [-1, 1]
  * industry = first two SIC digits (sic2 column is already that)

Output: <cache_dir>/panel.parquet
  permno, ym, sic2, mvel1, rf, ret, ret_ex, y1, y12, y12_ex, <94 ranked chars>
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Timer, ensure_dir, load_config, log

CHARS = [
    "mvel1", "beta", "betasq", "chmom", "dolvol", "idiovol", "indmom", "mom1m",
    "mom6m", "mom12m", "mom36m", "pricedelay", "turn", "absacc", "acc", "age",
    "agr", "bm", "bm_ia", "cashdebt", "cashpr", "cfp", "cfp_ia", "chatoia",
    "chcsho", "chempia", "chinv", "chpmia", "convind", "currat", "depr", "divi",
    "divo", "dy", "egr", "ep", "gma", "grcapx", "grltnoa", "herf", "hire",
    "invest", "lev", "lgr", "mve_ia", "operprof", "orgcap", "pchcapx_ia",
    "pchcurrat", "pchdepr", "pchgm_pchsale", "pchquick", "pchsale_pchinvt",
    "pchsale_pchrect", "pchsale_pchxsga", "pchsaleinv", "pctacc", "ps", "quick",
    "rd", "rd_mve", "rd_sale", "realestate", "roic", "salecash", "saleinv",
    "salerec", "secured", "securedind", "sgr", "sin", "sp", "tang", "tb",
    "aeavol", "cash", "chtx", "cinvest", "ear", "nincr", "roaq", "roavol",
    "roeq", "rsup", "stdacc", "stdcf", "ms", "baspread", "ill", "maxret",
    "retvol", "std_dolvol", "std_turn", "zerotrade",
]


def _load_crsp(paths: dict, dcfg: dict) -> pd.DataFrame:
    """Load the raw CRSP monthly returns and map the month index to YYYYMM.

    The file stores an integer month counter produced by the instructor with

        md = year(date) * 12 + month(date) - 1925 * 12 - 11

    so md = 1 is 1925-12.  Inverting it gives

        year*12 + month = md + 1925*12 + 11 = (1925*12 + 11) + md - 1

    which is what the arithmetic below does with base = 1925*12 + 11.
    `audit_md_rf.py` checks this decoding against the instructor's formula for
    every one of the 1,156 distinct md values in the file (0 disagreements), and
    independently against mom1m-era returns (offset 0, correlation 0.919).
    """
    p = paths["crsp_csv"]
    origin = dcfg.get("md_origin", [1925, 12])
    cr = pd.read_csv(p, usecols=["permno", "md", "retadj"],
                     dtype={"permno": "float32", "md": "int32", "retadj": "float32"})
    cr["permno"] = cr["permno"].astype("int32")
    base = int(origin[0]) * 12 + (int(origin[1]) - 1)
    v = base + (cr["md"].astype("int64") - 1)
    yr, mo = v // 12, v % 12 + 1
    cr["ym"] = (yr * 100 + mo).astype("int64")
    cr["ret"] = cr["retadj"].astype("float32")
    cr = cr.dropna(subset=["ret"]).drop_duplicates(["permno", "ym"])
    log(f"[crsp] {len(cr):,} return rows, months {cr['ym'].min()}..{cr['ym'].max()}")
    return cr[["permno", "ym", "ret"]]


def _fwd_compound(values: np.ndarray, horizon: int) -> np.ndarray:
    """y[i] = prod(1+values[i:i+horizon]) - 1 ; NaN where the window is incomplete."""
    t = len(values)
    out = np.full(t, np.nan, dtype=np.float64)
    if t < horizon:
        return out
    c = np.log1p(values)
    cs = np.concatenate(([0.0], np.cumsum(c)))
    out[: t - horizon + 1] = np.exp(cs[horizon:] - cs[: t - horizon + 1]) - 1.0
    return out


def build_panel(cfg: dict, force: bool = False) -> pd.DataFrame:
    paths = cfg["paths"]
    cache = ensure_dir(paths["cache_dir"])
    out_path = cache / "panel.parquet"
    import os

    if os.path.exists(out_path) and not force:
        log(f"[panel] reusing {out_path}")
        return pd.read_parquet(out_path)

    dcfg = cfg["data"]
    horizon = int(dcfg["horizon_months"])
    usecols = ["permno", "DATE"] + CHARS + ["sic2"]

    with Timer("read datashare.csv"):
        dtypes = {c: "float32" for c in CHARS}
        dtypes["permno"] = "int32"
        dtypes["DATE"] = "int32"
        dtypes["sic2"] = "float32"
        df = pd.read_csv(paths["raw_csv"], usecols=usecols, dtype=dtypes)
        log(f"[read] {len(df):,} rows x {df.shape[1]} cols  ({df.memory_usage().sum()/2**30:.2f} GiB)")

    df["ym"] = (df["DATE"] // 100).astype("int32")
    df = df.drop(columns=["DATE"])
    df = df[(df["ym"] >= dcfg["start_ym"]) & (df["ym"] <= dcfg["end_ym"])].copy()
    log(f"[filter] {len(df):,} rows after {dcfg['start_ym']}..{dcfg['end_ym']}")

    df.sort_values(["permno", "ym"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    # --- monthly total return ---------------------------------------------
    source = str(dcfg.get("return_source", "mom1m")).lower()
    if source == "crsp":
        crsp = _load_crsp(paths, dcfg)
        df = df.merge(crsp, on=["permno", "ym"], how="left")
        n_bad = int(df["ret"].isna().sum())
        log(f"[ret] CRSP retadj matched {len(df) - n_bad:,} / {len(df):,} rows "
            f"({n_bad:,} unmatched, {100 * n_bad / max(len(df), 1):.1f}%)")
    else:
        df["ret"] = df.groupby("permno", sort=False)["mom1m"].shift(dcfg["mom_lag_shift"])
        n_bad = int(df["ret"].isna().sum())
        log(f"[ret] {n_bad:,} rows without a usable next-month mom1m")

    # --- risk free & excess return ----------------------------------------
    macro = pd.read_csv(paths["macro_csv"])
    macro["ym"] = macro["ym"].astype("int32")
    rf_map = dict(zip(macro["ym"], macro["RF"]))
    df["rf"] = df["ym"].map(rf_map).astype("float32")
    df["ret_ex"] = df["ret"] - df["rf"]
    # keep the raw market cap: the rank transform below destroys the level,
    # but we need it for value weights and for the Top/Bottom-1000 rows of Table 2
    df["mvel1_raw"] = df["mvel1"].astype("float32")

    # --- horizons ----------------------------------------------------------
    # annual target = realised excess return over the next `horizon` months:
    #   prod(1+R_t..R_{t+h-1}) - prod(1+RF_t..RF_{t+h-1})
    y12 = np.full(len(df), np.nan)
    permno = df["permno"].to_numpy()
    gross = df["ret"].to_numpy(dtype=np.float64)
    bounds = np.flatnonzero(np.diff(permno)) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(df)]):
        y12[a:b] = _fwd_compound(gross[a:b], horizon)

    # 12-month cumulative risk-free over the same calendar window
    ym_now = df["ym"].to_numpy()
    months = np.sort(macro["ym"].unique())
    rf_series = macro.sort_values("ym")["RF"].to_numpy(dtype=np.float64)
    idx = {m: i for i, m in enumerate(months)}
    pos = np.array([idx[m] for m in ym_now])
    rf12 = np.full(len(df), np.nan)
    ok = pos <= len(months) - horizon
    cs = np.concatenate(([0.0], np.cumsum(np.log1p(rf_series))))
    rf12[ok] = np.exp(cs[pos[ok] + horizon] - cs[pos[ok]]) - 1.0

    df["rf12"] = rf12.astype("float32")
    df["y1"] = df["ret_ex"].astype("float32")
    df["y12"] = (y12 - rf12).astype("float32")

    df = df.dropna(subset=["ret"]).copy()

    # --- characteristics: median impute then rank into [-1,1] -------------
    with Timer("rank-normalise 94 characteristics"):
        by_month = df.groupby("ym", sort=False)
        for c in CHARS:
            v = df[c]
            med = v.groupby(df["ym"]).transform("median")
            v = v.fillna(med)
            r = v.groupby(df["ym"]).rank(method="average", pct=True)
            df[c] = (r * 2.0 - 1.0).astype("float32")
        del by_month
        # a few months can have an entirely missing characteristic -> rank is NaN
        df[CHARS] = df[CHARS].fillna(0.0)

    df["sic2"] = df["sic2"].fillna(-1).astype("int32")

    keep = ["permno", "ym", "sic2", "mvel1_raw", "rf", "ret", "ret_ex", "y1", "y12"] + CHARS
    panel = df[keep].copy()
    panel.to_parquet(out_path, index=False, compression="zstd")
    log(f"[panel] wrote {out_path}  ({len(panel):,} rows)")

    diag = {
        "rows": int(len(panel)),
        "stocks": int(panel["permno"].nunique()),
        "months": int(panel["ym"].nunique()),
        "stocks_per_month": float(panel.groupby("ym").size().mean()),
        "ym_min": int(panel["ym"].min()),
        "ym_max": int(panel["ym"].max()),
        "mean_ret_ex": float(panel["ret_ex"].mean()),
        "share_missing_y12": float(panel["y12"].isna().mean()),
    }
    log(f"[diag] {diag}")
    pd.Series(diag).to_json(cache / "panel_diag.json", force_ascii=False)
    return panel


def main() -> int:
    cfg = load_config()
    build_panel(cfg, force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
