"""Can the network even fit the TRAINING data?

In-sample IC vs out-of-sample IC for one refit.
  in-sample high, OOS low  -> overfitting (model/training issue)
  in-sample low too        -> the network is not learning at all (code bug)
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import nn as nnmod  # noqa: E402
from src.common import load_config  # noqa: E402
from src.features import load_meta, year_path  # noqa: E402
from src.train import target_y, year_bounds  # noqa: E402

TY = 2012


def within_month_ic(p, y, ym):
    import pandas as pd
    df = pd.DataFrame({"ym": ym, "p": p, "y": y})
    ic = df.groupby("ym").apply(lambda g: g.p.rank().corr(g.y.rank()), include_groups=False)
    return float(ic.mean()), float(ic.median()), float((ic > 0).mean())


def main():
    cfg = load_config()
    meta = load_meta(cfg)
    bounds = year_bounds(meta)
    y_all = target_y(meta, cfg)
    mcfg = cfg["model"]

    ty = TY
    train_years = list(range(1957, ty - 12))
    valid_years = list(range(ty - 12, ty))
    y_by_year = {y: y_all[a:b] for y, (a, b) in bounds.items()}
    ym_by_year = {y: meta["ym"].to_numpy()[a:b] for y, (a, b) in bounds.items()}

    def loader(year):
        return np.load(year_path(cfg, year), mmap_mode="r")[:, :94]

    model, info = nnmod.fit(
        loader, y_by_year, train_years, valid_years,
        hidden=mcfg["hidden"], hp=mcfg, seed=1, n_in=94,
        ym_by_year=ym_by_year, log_prefix="[audit]", max_epochs=30,
    )
    scale = float(mcfg["target_scale"])
    tm = float(info["train_mean"])

    def ic_on(years, tag):
        ps, ys, ms = [], [], []
        for y in years:
            a, b = bounds[y]
            X = np.asarray(loader(y))
            yy = y_by_year[y]
            ok = np.isfinite(yy)
            ps.append(nnmod._predict_array(model, X[ok]) / scale)
            ys.append(yy[ok] - tm)
            ms.append(ym_by_year[y][ok])
        p, t, m = (np.concatenate(z) for z in (ps, ys, ms))
        mean, med, share = within_month_ic(p, t, m)
        print(f"  {tag:26s} rows={len(p):>9,}  IC mean={mean:+.4f} median={med:+.4f} "
              f"share>0={100*share:.0f}%  pred sd={p.std():.4f}")
        return mean

    print(f"refit for test year {ty}: train 1957-{ty-13}, valid {valid_years[0]}-{valid_years[-1]}")
    ic_tr = ic_on(train_years, "IN-SAMPLE (train years)")
    ic_va = ic_on(valid_years, "validation")
    ic_te = ic_on([ty], "OUT-OF-SAMPLE (test)")
    print(f"\n  in-sample {ic_tr:+.4f}  ->  OOS {ic_te:+.4f}   ratio {ic_te/max(ic_tr,1e-9):.2f}")


if __name__ == "__main__":
    main()
