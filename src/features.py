"""Assemble the 920-column design matrix  z = [c , c (x) x , industry dummies].

    c  : 94 rank-normalised characteristics          -> 94
    x  : 8 Welch-Goyal macro predictors, expanding z-scored (no look-ahead)
    z  : c plus c interacted with each macro var     -> 94 * 9 = 846
    ind: top-74 two-digit SIC dummies                -> 74
    total = 920  (paper p.2249: 94*(8+1)+74 = 920)

Per-year float32 blocks go to <cache>/z/<year>.npy; row meta to <cache>/z/meta.parquet.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Timer, ensure_dir, load_config, log
from .prep import CHARS

MACRO_VARS = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]
N_CHARS = len(CHARS)
N_MACRO = len(MACRO_VARS)


def _macro_z_matrix(macro: pd.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    """Expanding z-score of each macro predictor -> (n_months, 8) + ym->row index.

    Expanding (rather than full-sample) moments keep the transform point-in-time.
    """
    macro = macro.sort_values("ym").reset_index(drop=True)
    cols = []
    for v in MACRO_VARS:
        s = macro[v].astype(float)
        mean = s.expanding(min_periods=24).mean()
        std = s.expanding(min_periods=24).std().replace(0.0, np.nan)
        cols.append(((s - mean) / std).fillna(0.0).to_numpy(dtype=np.float32))
    mat = np.column_stack(cols)
    idx = {int(m): i for i, m in enumerate(macro["ym"].to_numpy())}
    return mat, idx


def build_design(cfg: dict, panel: pd.DataFrame, force: bool = False) -> str:
    cache = ensure_dir(cfg["paths"]["cache_dir"])
    zdir = ensure_dir(cache / "z")
    done_flag = zdir / "_DONE"
    meta_path = zdir / "meta.parquet"
    if done_flag.exists() and not force:
        log("[design] cached, reuse")
        return str(zdir)

    dcfg = cfg["data"]
    keep_end = int(dcfg.get("model_end_ym", dcfg["end_ym"]))
    panel = panel[panel["ym"] <= keep_end].copy()
    log(f"[design] modelling rows {len(panel):,}  ({panel['ym'].min()}..{panel['ym'].max()})")

    macro = pd.read_csv(cfg["paths"]["macro_csv"])
    mz_mat, mz_idx = _macro_z_matrix(macro)

    # ---- industry dummies: top-74 sic2 codes over the modelling sample ----
    n_ind = int(dcfg["n_industries"])
    top = panel["sic2"].value_counts().head(n_ind).index.to_numpy()
    ind_map = {int(c): i for i, c in enumerate(top)}
    ind_code = panel["sic2"].map(ind_map).fillna(-1).to_numpy(dtype=np.int16)
    log(f"[design] industry: kept {len(ind_map)} sic2 codes")

    order = np.argsort(panel["ym"].to_numpy(), kind="mergesort")
    panel = panel.iloc[order].reset_index(drop=True)
    ind_code = ind_code[order]

    ym_arr = panel["ym"].to_numpy()
    month_pos = np.array([mz_idx[int(m)] for m in ym_arr], dtype=np.int32)

    meta = panel[["permno", "ym", "sic2", "mvel1_raw", "y1", "y12"]].copy()
    meta["ind_code"] = ind_code
    meta.to_parquet(meta_path, index=False)

    years = ym_arr // 100
    uniq = np.unique(years)
    c_all = panel[CHARS].to_numpy(dtype=np.float32)
    del panel

    with Timer("build per-year design blocks"):
        pos = 0
        for y in uniq:
            n = int((years == y).sum())
            sl = slice(pos, pos + n)
            c = c_all[sl]
            m = mz_mat[month_pos[sl]]
            blocks = [c] + [c * m[:, k : k + 1] for k in range(N_MACRO)]
            ind = ind_code[sl]
            d = np.zeros((n, n_ind), dtype=np.float32)
            valid = ind >= 0
            d[np.arange(n)[valid], ind[valid]] = 1.0
            Z = np.concatenate(blocks + [d], axis=1)
            assert Z.shape[1] == N_CHARS * 9 + n_ind, Z.shape
            np.save(zdir / f"{y}.npy", Z)
            pos += n
            print(f"    {y}: {n:>7,} rows  {Z.shape}  {Z.nbytes/2**20:.0f} MiB", flush=True)

    done_flag.write_text("ok\n", encoding="utf-8")
    log(f"[design] wrote {zdir}")
    return str(zdir)


def load_meta(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(Path(cfg["paths"]["cache_dir"]) / "z" / "meta.parquet")


def year_path(cfg: dict, year: int) -> Path:
    return Path(cfg["paths"]["cache_dir"]) / "z" / f"{year}.npy"


def main() -> int:
    from .prep import build_panel

    cfg = load_config()
    panel = build_panel(cfg, force=False)
    build_design(cfg, panel, force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
