"""Hand-verify y12 against a manual product of raw CRSP monthly returns.

This is the one step in the pipeline that audit_synth.py cannot cover, because the
synthetic test injects its own target.  If the 12-month compounding, the window
placement, or the risk-free subtraction is wrong, every annual number is wrong
while the code still 'works'.
"""
import numpy as np
import pandas as pd

PANEL = r"D:\DSHworkspace\DSH操作田\dsh空白操作田\gkx_data\cache_crsp\panel.parquet"
CRSP = r"D:\DSHworkspace\DSH操作田\dsh空白操作田\gkx_data\crsp_ret_all.csv"

pn = pd.read_parquet(PANEL, columns=["permno", "ym", "ret", "rf", "ret_ex", "y12"])

# pick a few permnos with long uninterrupted histories
cnt = pn.groupby("permno").size()
cands = cnt[cnt > 500].index[:3]

for p in cands:
    sub = pn[pn.permno == p].sort_values("ym").reset_index(drop=True)
    ok = sub[sub.y12.notna()]
    if len(ok) < 3:
        continue
    i = len(ok) // 2
    row = ok.iloc[i]
    pos = sub.index[sub.ym == row.ym][0]
    win = sub.iloc[pos: pos + 12]
    if len(win) < 12:
        continue
    gross = float(np.prod(1.0 + win.ret.to_numpy()))
    rf12 = float(np.prod(1.0 + win.rf.to_numpy()))
    manual = gross - rf12
    print(f"\npermno {int(p)}  from {int(win.ym.iloc[0])} to {int(win.ym.iloc[-1])} "
          f"({len(win)} months)")
    print(f"  panel y12      = {row.y12:+.6f}")
    print(f"  manual product = {manual:+.6f}")
    print(f"  difference     = {row.y12 - manual:+.3e}")
    print(f"  monthly rets   = {np.round(win.ret.to_numpy()[:6], 4)} ...")
    print(f"  rf sum over 12 = {win.rf.sum():.5f}")
