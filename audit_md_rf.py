"""Verify the two data conventions supplied by the instructor.

1. `md` in crsp_ret_all.csv is a processed month index with

       md = year(date) * 12 + month(date) - 1925 * 12 - 11

   so md = 1 is 1925-12.  This script round-trips the formula against the
   decoding used in src/prep.py for every month present in the file.

2. The risk-free rate is the official Fama-French series from
   https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

   The script compares the RF column used by the panel against
   F-F_Research_Data_Factors, month by month.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r"D:\DSHworkspace\DSH操作田\dsh空白操作田\gkx_data")
CRSP = DATA / "crsp_ret_all.csv"
FF_ZIP = DATA / "F-F_Research_Data_Factors_CSV.zip"
MACRO = Path(r"D:\DSHworkspace\DSH操作田\dsh空白操作田\data\gkx_macro_panel.csv")

BASE = 1925 * 12 + 11


def md_from_ym(ym: int) -> int:
    y, m = divmod(ym, 100)
    return y * 12 + m - BASE


def ym_from_md_ours(md: int) -> int:
    v = BASE + (md - 1)
    return (v // 12) * 100 + (v % 12 + 1)


def ym_from_md_instructor(md: int) -> int:
    v = md + BASE
    y = (v - 1) // 12
    m = (v - 1) % 12 + 1
    return y * 100 + m


def main() -> None:
    print("=" * 62)
    print("1. md  <->  year-month")
    print("=" * 62)
    for ym in (192512, 195703, 196307, 199904, 200807, 201612):
        md = md_from_ym(ym)
        a, b = ym_from_md_ours(md), ym_from_md_instructor(md)
        flag = "OK" if a == b == ym else "MISMATCH"
        print(f"  {ym}: md={md:5d}  ours={a}  instructor={b}  {flag}")

    md = pd.read_csv(CRSP, usecols=["md"])["md"].to_numpy()
    uniq = np.unique(md)
    bad = [int(m) for m in uniq
           if ym_from_md_ours(int(m)) != ym_from_md_instructor(int(m))]
    print(f"\n  distinct md in file: {len(uniq):,}  "
          f"range [{uniq.min()}, {uniq.max()}]")
    print(f"  months where the two decodings disagree: {len(bad)}")

    print()
    print("=" * 62)
    print("2. risk-free rate vs Fama-French")
    print("=" * 62)
    raw = zipfile.ZipFile(FF_ZIP).read(
        zipfile.ZipFile(FF_ZIP).namelist()[0]).decode("latin-1")
    rows = [[p.strip() for p in ln.strip().split(",")][:5]
            for ln in raw.splitlines() if re.fullmatch(r"\d{6}.*", ln.strip())]
    ff = pd.DataFrame(rows, columns=["ym", "MktRF", "SMB", "HML", "RF"])
    ff["ym"] = ff["ym"].astype(int)
    ff = ff[ff.ym.between(195703, 201612)].copy()
    for c in ("MktRF", "SMB", "HML", "RF"):
        ff[c] = pd.to_numeric(ff[c], errors="coerce") / 100.0

    ours = pd.read_csv(MACRO)[["ym", "RF", "MktRF"]]
    m = ff.merge(ours, on="ym", suffixes=("_ff", "_ours"))
    d_rf = (m["RF_ff"] - m["RF_ours"]).abs()
    d_mk = (m["MktRF_ff"] - m["MktRF_ours"]).abs()
    print(f"  overlapping months      : {len(m):,}")
    print(f"  RF   max abs difference : {d_rf.max():.3e}")
    print(f"  RF   identical (<1e-9)  : {100*(d_rf < 1e-9).mean():.1f}%")
    print(f"  MktRF max abs difference: {d_mk.max():.3e}")
    print(f"  RF monthly mean (FF)    : {m['RF_ff'].mean():.6f}")


if __name__ == "__main__":
    main()
