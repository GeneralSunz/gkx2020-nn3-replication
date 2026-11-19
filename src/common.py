"""Shared helpers: config loading, paths, logging."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../gkx_nn
PROJECT = ROOT.parent                      # workspace root


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path or os.environ.get("GKX_CONFIG") or (ROOT / "config.json"))
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # resolve relative paths against project root
    for k, v in list(cfg["paths"].items()):
        pv = Path(v)
        cfg["paths"][k] = str(pv if pv.is_absolute() else (PROJECT / pv))
    cfg["paths"]["cache_dir"] = str(Path(cfg["paths"]["cache_dir"]))
    cfg["paths"]["out_dir"] = str(Path(cfg["paths"]["out_dir"]))
    return cfg


class Timer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = time.time()

    def __enter__(self):
        log(f"[start] {self.label}")
        return self

    def __exit__(self, *exc):
        log(f"[done ] {self.label}  ({time.time() - self.t0:.1f}s)")


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"{ts} {msg}", flush=True)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def hard_exit(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)
