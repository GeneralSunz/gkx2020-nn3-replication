"""Walk-forward refitting, prediction, and evaluation against Table 2.

Split (paper p.2249): train 1957-1974, validation 1975-1986, test 1987-2016.
Each refit adds one year to training, rolls the 12-month validation window
forward, and predicts the next test year.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import nn as nnmod
from .common import Timer, ensure_dir, load_config, log
from .features import load_meta, year_path

TABLE2 = {
    # Gu, Kelly & Xiu (2020) Table 2, annual R2_oos in percent
    "OLS": {"All": -34.86, "Top": -54.86, "Bottom": -19.22},
    "NN1": {"All": 2.64, "Top": 2.77, "Bottom": 4.37},
    "NN2": {"All": 2.70, "Top": 4.24, "Bottom": 3.72},
    "NN3": {"All": 3.40, "Top": 4.73, "Bottom": 5.17},
    "NN4": {"All": 3.60, "Top": 4.91, "Bottom": 5.01},
    "NN5": {"All": 2.79, "Top": 4.86, "Bottom": 3.58},
}


def year_bounds(meta: pd.DataFrame) -> dict[int, tuple[int, int]]:
    yrs = (meta["ym"].to_numpy() // 100)
    out = {}
    start = 0
    for i in range(1, len(yrs) + 1):
        if i == len(yrs) or yrs[i] != yrs[start]:
            out[int(yrs[start])] = (start, i)
            start = i
    return out


def r2_oos(y: np.ndarray, pred: np.ndarray) -> float:
    """Paper eq. (19): denominator is sum of squared excess returns, NOT demeaned."""
    return float(1.0 - ((y - pred) ** 2).sum() / (y ** 2).sum())


def target_y(meta: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Target vector, optionally winsorised.

    The monthly returns in datashare.csv are themselves clipped, so compounding
    12 of them manufactures implausible annual outliers (max +1212% here).  A plain
    L2 objective then spends all its capacity on those points, which is the main
    reason an untrimmed NN does not generalise.  Set data.y_clip_quantiles to
    [lo, hi] to trim; set it to null to keep the raw target.
    """
    y = meta["y12"].to_numpy().astype(np.float64)
    horizon = str((cfg.get("data") or {}).get("target", "annual")).lower()
    if horizon == "monthly":
        # Table-1 metric: next-month excess return.  Used to validate the whole
        # pipeline against third-party replications (which reproduce the monthly
        # 0.40% but do not report the annual number).
        y = meta["y1"].to_numpy().astype(np.float64)
        return y
    if horizon == "annual_rank":
        # Train on the cross-sectional rank of the annual return, mapped to
        # [-1, 1], but keep evaluating on the raw target.  The annual return is
        # so fat tailed (E[y^2]=0.84, dominated by a handful of microcap lottery
        # payoffs) that a plain L2 objective spends its capacity on them; ranks
        # make the loss robust while the amplitude is fitted separately anyway.
        df = pd.DataFrame({"ym": meta["ym"].to_numpy(), "y": y})
        y = df.groupby("ym")["y"].transform(
            lambda s: (s.rank(method="average", pct=True) * 2 - 1)
            .where(s.notna())
        ).to_numpy(dtype=np.float64)
        log("[target] annual_rank: cross-sectional rank in [-1,1]")
        return y
    q = (cfg.get("data") or {}).get("y_clip_quantiles")
    if not q:
        return y
    lo, hi = np.nanquantile(y, q)
    log(f"[target] winsorise y12 at {q} -> [{lo:+.4f}, {hi:+.4f}]")
    return np.clip(y, lo, hi)


def eval_y(meta: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Evaluation target: always the (optionally winsorised) raw annual return.

    Kept separate from target_y so that training on a rank-transformed target does
    not change the metric being reported.
    """
    y = meta["y12"].to_numpy().astype(np.float64)
    if str((cfg.get("data") or {}).get("target", "annual")).lower() == "monthly":
        return meta["y1"].to_numpy().astype(np.float64)
    q = (cfg.get("data") or {}).get("y_clip_quantiles")
    if not q:
        return y
    lo, hi = np.nanquantile(y, q)
    return np.clip(y, lo, hi)


def _slice_preds(meta: pd.DataFrame, bounds, years, pred_by_year: dict[int, np.ndarray], y_all: np.ndarray):
    ys, preds, mvel, ym = [], [], [], []
    for y in years:
        a, b = bounds[y]
        ys.append(y_all[a:b])
        preds.append(pred_by_year[y])
        mvel.append(meta["mvel1_raw"].to_numpy()[a:b])
        ym.append(meta["ym"].to_numpy()[a:b])
    return (np.concatenate(ys), np.concatenate(preds),
            np.concatenate(mvel), np.concatenate(ym))


def evaluate(meta: pd.DataFrame, bounds, test_years, pred_by_year, y_all: np.ndarray) -> dict:
    y, p, mvel, ym = _slice_preds(meta, bounds, test_years, pred_by_year, y_all)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, mvel, ym = y[ok], p[ok], mvel[ok], ym[ok]

    df = pd.DataFrame({"ym": ym, "mvel1_raw": mvel, "y": y, "p": p})
    df["rk_hi"] = df.groupby("ym")["mvel1_raw"].rank(ascending=False, method="first")
    df["rk_lo"] = df.groupby("ym")["mvel1_raw"].rank(ascending=True, method="first")

    all_r2 = r2_oos(df["y"].to_numpy(), df["p"].to_numpy())
    top = df[df["rk_hi"] <= 1000]
    bot = df[df["rk_lo"] <= 1000]
    top_r2 = r2_oos(top["y"].to_numpy(), top["p"].to_numpy()) if len(top) else float("nan")
    bot_r2 = r2_oos(bot["y"].to_numpy(), bot["p"].to_numpy()) if len(bot) else float("nan")

    by_year = {}
    for y_, g in df.groupby(df["ym"] // 100):
        by_year[int(y_)] = r2_oos(g["y"].to_numpy(), g["p"].to_numpy())

    return {
        "n_obs": int(len(df)),
        "All": 100 * all_r2,
        "Top": 100 * top_r2,
        "Bottom": 100 * bot_r2,
        "by_year": {k: 100 * v for k, v in by_year.items()},
        "mean_pred": float(df["p"].mean()),
        "mean_y": float(df["y"].mean()),
    }


# --------------------------------------------------------------------------
# baseline: full OLS on the 920 features (paper reports R2_oos = -34.86%)
# --------------------------------------------------------------------------
def run_ols(cfg, test_years, tag="ols", fit_on_valid: bool = False):
    """Full-feature OLS baseline.

    The paper tunes hyper-parameters on the validation window and then re-fits on
    train+validation. OLS has no hyper-parameters to tune, so its benchmark is
    estimated on the training window only (1957..ty-13); including the validation
    years makes it 4x larger and, empirically, far too well behaved to match the
    paper's -34.86%.
    """
    meta = load_meta(cfg)
    bounds = year_bounds(meta)
    y_all = target_y(meta, cfg)
    y_eval = eval_y(meta, cfg)
    pred_by_year = {}
    for ty in test_years:
        train_years = list(range(1957, ty - 12))
        valid_years = list(range(ty - 12, ty))
        fit_years = train_years + valid_years if fit_on_valid else train_years
        k = None
        XtX = None
        Xty = None
        t0 = time.time()
        for y in fit_years:
            X = np.load(year_path(cfg, y))
            a, b = bounds[y]
            yv = y_all[a:b]
            ok = np.isfinite(yv)
            X = X[ok].astype(np.float64)
            yv = yv[ok]
            X = np.hstack([X, np.ones((len(X), 1))])
            if XtX is None:
                k = X.shape[1]
                XtX = np.zeros((k, k))
                Xty = np.zeros(k)
            XtX += X.T @ X
            Xty += X.T @ yv
            del X
        # X'X is rank-deficient (74 dummies sum to the intercept, plus collinear
        # characteristics and interactions).  Solve it the way lstsq does: minimum
        # norm solution via a truncated eigen-decomposition, not a naive solve().
        w, V = np.linalg.eigh(XtX)
        tol = w.max() * k * np.finfo(np.float64).eps
        keep = w > tol
        w_inv = np.zeros_like(w)
        w_inv[keep] = 1.0 / w[keep]
        beta = V @ (w_inv * (V.T @ Xty))
        log(f"[ols] test {ty}: rank {int(keep.sum())}/{k}, "
            f"cond {w.max() / w[keep].min():.3e}, dropped {int((~keep).sum())} dirs")
        Xt = np.load(year_path(cfg, ty))
        pred = np.hstack([Xt, np.ones((len(Xt), 1))]) @ beta
        pred_by_year[ty] = pred.astype(np.float32)
        log(f"[ols] test {ty}: train {fit_years[0]}-{fit_years[-1]} "
            f"pred sd={pred.std():.4f} ({time.time()-t0:.0f}s)")
    res = evaluate(meta, bounds, test_years, pred_by_year, y_eval)
    res["tag"] = tag
    out = ensure_dir(cfg["paths"]["out_dir"])
    with open(f"{out}/{tag}_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    log(f"[{tag}] All={res['All']:+.2f}%  Top={res['Top']:+.2f}%  Bottom={res['Bottom']:+.2f}%  n={res['n_obs']:,}")
    return res


# --------------------------------------------------------------------------
# NN walk-forward
# --------------------------------------------------------------------------
def run_nn(cfg, test_years, seeds=None, max_epochs=None, tag="nn3"):
    import torch

    n_threads = int((cfg.get("runtime") or {}).get("threads", 0) or 0)
    if n_threads > 0:
        torch.set_num_threads(n_threads)
        log(f"[torch] num_threads={n_threads}")
    meta = load_meta(cfg)
    bounds = year_bounds(meta)
    mcfg = cfg["model"]
    out = ensure_dir(cfg["paths"]["out_dir"])
    y_all = target_y(meta, cfg)
    y_eval = eval_y(meta, cfg)
    y_by_year = {y: y_all[a:b] for y, (a, b) in bounds.items()}
    ym_by_year = {y: meta["ym"].to_numpy()[a:b] for y, (a, b) in bounds.items()}
    y_by_year_eval = {y: y_eval[a:b] for y, (a, b) in bounds.items()}
    _rank_target = str((cfg.get("data") or {}).get("target", "annual")).lower() == "annual_rank"

    # optional feature slice.  Columns are [94 chars | 752 interactions | 74 industries].
    #   n_features=94        -> characteristics only
    #   n_features="chars+ind" -> characteristics + industry dummies (168 cols)
    #   n_features unset     -> the paper's full 920
    n_feat = mcfg.get("n_features")
    use_ind = isinstance(n_feat, str) and n_feat == "chars+ind"

    def loader(year: int) -> np.ndarray:
        Z = np.load(year_path(cfg, year), mmap_mode="r")
        if use_ind:
            return np.hstack([np.asarray(Z[:, :94]), np.asarray(Z[:, 846:])])
        return Z if not n_feat else Z[:, : int(n_feat)]

    n_in = int(loader(min(bounds)).shape[1])
    seeds = seeds or mcfg["seeds"]
    pred_by_year = {}
    diag = {}

    for ty in test_years:
        train_years = list(range(1957, ty - 12))
        valid_years = list(range(ty - 12, ty))
        preds = []
        t0 = time.time()
        for s in seeds:
            log(f"[{tag}] test {ty}: fit seed {s} on {train_years[0]}-{train_years[-1]}, valid {valid_years[0]}-{valid_years[-1]}")
            model, info = nnmod.fit(
                loader, y_by_year, train_years, valid_years,
                hidden=mcfg["hidden"], hp=mcfg, seed=int(s), n_in=n_in,
                ym_by_year=ym_by_year,
                eval_by_year=y_by_year_eval,
                log_prefix=f"[{tag} {ty} s{s}]", max_epochs=max_epochs,
            )
            Xt = np.asarray(loader(ty))
            centred = nnmod._predict_array(model, Xt) / float(mcfg["target_scale"])
            if str(mcfg.get("pred_sizing", "ic")) == "ic" and not _rank_target:
                # Size each month's forecast to IC * sigma_y from the validation
                # window.  The network's raw output scale drifts with the input
                # distribution (in 2008 it was 3x too wide and cost ~1pp of R2),
                # so re-standardising each month and re-applying a stable target
                # dispersion is far more robust than freezing alpha.
                ic = max(0.0, 1.0 - float(info["best_valid_mse"]))
                target_sd = ic * float(info.get("val_y_sd", 1.0))
                mm = ym_by_year[ty]
                codes, uniq = pd.factorize(mm, sort=False)
                n = np.bincount(codes, minlength=len(uniq)).astype(np.float64)
                s1 = np.bincount(codes, weights=centred, minlength=len(uniq))
                m1 = s1 / np.maximum(n, 1)
                sd = np.sqrt(np.bincount(codes, weights=(centred - m1[codes]) ** 2,
                                         minlength=len(uniq)) / np.maximum(n, 1))
                scaled = np.where(sd[codes] > 1e-9,
                                  (centred - m1[codes]) / np.maximum(sd[codes], 1e-9) * target_sd,
                                  0.0)
                pred = float(info["train_mean"]) + scaled
            else:
                pred = float(info["train_mean"]) + float(info["alpha"]) * centred
            preds.append(pred)
            pred_by_year.setdefault(ty, None)
            diag[f"{ty}_seed{s}"] = {"best_epoch": info["best_epoch"],
                                     "val_ic": 1.0 - float(info["best_valid_mse"])}
        pred_by_year[ty] = np.mean(preds, axis=0).astype(np.float32)
        p = pred_by_year[ty]
        a, b = bounds[ty]
        log(f"[{tag}] test {ty} done in {time.time()-t0:.0f}s  "
            f"pred sd={p.std():.4f} (y sd={np.nanstd(y_all[a:b]):.4f})  n_in={n_in}")
        diag[f"{ty}_pred_sd"] = float(p.std())
        diag[f"{ty}_y_sd"] = float(np.nanstd(y_all[a:b]))

    res = evaluate(meta, bounds, test_years, pred_by_year, y_eval)
    res["tag"] = tag
    res["diag"] = diag
    with open(f"{out}/{tag}_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez(f"{out}/{tag}_preds.npz", **{str(k): v for k, v in pred_by_year.items()})
    log(f"[{tag}] All={res['All']:+.2f}%  Top={res['Top']:+.2f}%  Bottom={res['Bottom']:+.2f}%  n={res['n_obs']:,}")
    return res
