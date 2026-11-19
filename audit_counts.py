"""Why is our monthly stock count 5,206 when the paper says >6,200?

Two candidates:
  (a) the authors' characteristics file is a filtered subset of their return
      universe, so the paper's 6,200 counts stocks without characteristics;
  (b) we are losing rows in the CRSP merge.

This counts, month by month over 1957-03..2016-12:
   - permno-months in the CRSP return file
   - permno-months in datashare.csv
   - permno-months in our panel
"""
import numpy as np
import pandas as pd

CRSP = r"D:\DSHworkspace\DSH操作田\dsh空白操作田\gkx_data\crsp_ret_all.csv"
DATA = r"D:\DSHworkspace\DSH操作田\dsh空白操作田\gkx_data\datashare.csv"
PANEL = r"D:\DSHworkspace\DSH操作田\dsh空白操作田\gkx_data\cache_crsp\panel.parquet"

LO, HI = 195703, 201612

# --- CRSP ---
cnt_crsp = {}
for ch in pd.read_csv(CRSP, usecols=["permno", "md"], chunksize=3_000_000):
    v = 1925 * 12 + 11 + (ch["md"].to_numpy() - 1)
    ym = (v // 12) * 100 + (v % 12 + 1)
    s = pd.Series(ym)
    s = s[(s >= LO) & (s <= HI)]
    for k, n in s.value_counts().items():
        cnt_crsp[k] = cnt_crsp.get(k, 0) + int(n)
print(f"CRSP months {len(cnt_crsp)}  rows {sum(cnt_crsp.values()):,}  "
      f"avg/month {np.mean(list(cnt_crsp.values())):,.0f}")

# --- datashare ---
ds = pd.read_csv(DATA, usecols=["permno", "DATE"], dtype={"DATE": "int64"})
ds["ym"] = ds["DATE"] // 100
ds = ds[ds.ym.between(LO, HI)]
c_ds = ds.groupby("ym").size()
print(f"datashare months {len(c_ds)}  rows {len(ds):,}  avg/month {c_ds.mean():,.0f}")

# --- panel ---
pn = pd.read_parquet(PANEL, columns=["ym"])
c_pn = pn.groupby("ym").size()
print(f"panel    months {len(c_pn)}  rows {len(pn):,}  avg/month {c_pn.mean():,.0f}")

common = c_ds.index.intersection(pd.Index(list(cnt_crsp.keys())))
cr = pd.Series({m: cnt_crsp[m] for m in common}).sort_index()
ds_s = c_ds.reindex(common).fillna(0).astype(int)
pn_s = c_pn.reindex(common).fillna(0).astype(int)
print(f"\non {len(common)} common months:")
print(f"  CRSP      {cr.mean():,.0f} / month")
print(f"  datashare {ds_s.mean():,.0f} / month   ({100*ds_s.sum()/cr.sum():.1f}% of CRSP rows)")
print(f"  panel     {pn_s.mean():,.0f} / month")
print(f"  paper     >6,200 / month")

print("\nsample months:")
idx = [common[i] for i in range(0, len(common), max(1, len(common)//8))]
for m in idx:
    print(f"  {m}: crsp={cr[m]:>5,}  datashare={ds_s[m]:>5,}  panel={pn_s[m]:>5,}")
