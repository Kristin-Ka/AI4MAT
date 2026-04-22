#!/usr/bin/env python3
"""Aggregate every ``results/<model>/<model>_<task>/metrics.json`` into a
single ``results/REPORT.md`` summary table.

Shows both original-unit metrics and transformed-space metrics (for
log1p targets), and flags runs whose test-set size indicates they were
trained on the legacy 7k dielectric subset instead of the full MP
summary dump (~150k).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _n_test(run_dir: Path) -> int | None:
    npz = run_dir / "predictions.npz"
    if not npz.exists():
        return None
    try:
        return int(np.load(npz)["y_true_transformed"].size)
    except Exception:
        return None


def main() -> int:
    rows = []
    for mp in sorted(RESULTS.rglob("metrics.json")):
        with open(mp) as f:
            m = json.load(f)
        run_dir = mp.parent
        # Layout is ``results/<model>/<arch>_<task>/metrics.json``.  The
        # top-level folder is the authoritative model identifier (so
        # ``xgboost`` vs ``xgboost_3d`` don't collide even though both
        # use ``arch=xgboost``).
        model = run_dir.parent.name
        _, _, task = run_dir.name.partition("_")
        if not task:
            task = run_dir.name
        orig = m.get("original_units", m)
        n_test = _n_test(run_dir)
        subset = "dielectric-7k" if n_test and n_test < 5000 else "MP-full"
        rows.append({
            "task": task,
            "model": model,
            "subset": subset,
            "N_test": n_test,
            "RMSE (orig)": round(orig.get("RMSE", float("nan")), 4),
            "MAE (orig)": round(orig.get("MAE", float("nan")), 4),
            "R² (orig)": round(orig.get("R2", float("nan")), 4),
            "R² (trained)": round(m.get("R2", float("nan")), 4),
            "trained in": m.get("training_space", "identity"),
        })
    if not rows:
        print("No results found.")
        return 0

    df = pd.DataFrame(rows).sort_values(["task", "model", "subset"]).reset_index(drop=True)
    md = [
        "# Benchmark Summary",
        "",
        "90:10 train/test split (matminer convention).  Descriptor models carve",
        "a small internal validation fold from train for XGBoost early stopping.",
        "",
        "* **subset=MP-full** — 154,879 MP materials from `mp_summary.jsonl`.",
        "* **subset=dielectric-7k** — 7,327-material dielectric subset (legacy).",
        "",
        tabulate(df, headers="keys", tablefmt="github", showindex=False),
    ]
    out = "\n".join(md) + "\n"
    (RESULTS / "REPORT.md").write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
