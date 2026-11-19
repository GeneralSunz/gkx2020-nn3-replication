"""Independent re-derivation of the rank normalisation.

Take one month straight from datashare.csv, recompute the cross-sectional rank of
a few characteristics by hand, and compare with what panel.parquet stores.  A
silent bug in prep.py (wrong groupby, wrong pct, wrong imputation) would poison
every downstream result and is invisible to model-side debugging.
"""
import numpy as np
import pandas as pd

RAW = r"D:\DSHworkspace\DSH操作田\dsh空白操作田\gkx_data\datashare.csv"
PANEL = r"D:\DSHworkspace\DSH操作田\dsh空白操作田\gkx_data\cache_crsp\panel.parquet"
CHECKS = ["mom1m", "mom12m", "bm", "retvol", "mvel1", "turn"]
YM = 200506

usecols = ["permno", "DATE"] + CHECKS
raw = pd.read_csv(RAW, usecols=usecols)
raw["ym"] = raw["DATE"] // 100
raw = raw[raw["ym"] == YM].copy()
print(f"raw rows for {YM}: {len(raw):,}")

panel = pd.read_parquet(PANEL, columns=["permno", "ym"] + CHECKS)
panel = panel[panel["ym"] == YM].copy()
print(f"panel rows for {YM}: {len(panel):,}")

m = raw[["permno"] + CHECKS].merge(panel[["permno"] + CHECKS], on="permno",
                                   suffixes=("_raw", "_panel"))
print(f"matched: {len(m):,}")

for c in CHECKS:
    v = m[f"{c}_raw"]
    # reproduce prep.py: median impute within the month, then rank(pct)*2-1
    v = v.fillna(v.median())
    r = v.rank(method="average", pct=True) * 2 - 1
    stored = m[f"{c}_panel"]
    diff = (r - stored).abs()
    print(f"  {c:8s} max|recomputed - stored| = {diff.max():.3e}   "
          f"stored range [{stored.min():+.2f}, {stored.max():+.2f}]")

# also verify the forward return for one permno against the raw CRSP file
CRSP = r"D:\DSHworkspace\DSH操作田\dsh空白操作田\gkx_data\crsp_ret_all.csv"
cr = pd.read_csv(CRSP, usecols=["permno", "md", "retadj"])
base = 1925 * 12 + 11
cr["ym"] = ((base + (cr["md"] - 1)) // 12) * 100 + ((base + (cr["md"] - 1)) % 12 + 1)
p = 10006
crs = cr[(cr.permno == p) & (cr.ym.between(195703, 195803))].sort_values("ym")
pn = pd.read_parquet(PANEL, columns=["permno", "ym", "ret"])
pns = pn[(pn.permno == p) & (pn.ym.between(195703, 195803))].sort_values("ym")
mm = pns.merge(crs, on="ym", suffixes=("_panel", "_crsp"))
print(f"\npermno {p}: panel ret vs crsp retadj")
print(mm[["ym", "ret_panel", "retadj"]].to_string(index=False, float_format=lambda x: f"{x:+.6f}"))
