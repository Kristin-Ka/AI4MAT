#!/usr/bin/env python3
"""Re-render all parity plots from stored ``predictions.npz`` files.

Use this after a plot-styling change (e.g. switching histogram marginals
to KDE curves) to refresh every model's ``parity_plot.png`` without
retraining.  Walks ``results/`` recursively and re-plots every folder
that contains a ``predictions.npz`` with both train and test arrays.

Usage:
    python scripts/rerender_parity_plots.py
    python scripts/rerender_parity_plots.py --results-dir results
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matprop_nn  # noqa: F401
from matprop_nn.evaluation.plots import parity_plot

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("rerender")


def rerender_one(pred_path: Path) -> bool:
    data = np.load(pred_path)
    if "y_pred_transformed" not in data.files or "y_true_transformed" not in data.files:
        logger.warning("  %s: missing transformed arrays, skipping.", pred_path)
        return False

    y_true = data["y_true_transformed"]
    y_pred = data["y_pred_transformed"]
    y_train_true = data.get("y_train_true_transformed") if hasattr(data, "get") else None
    y_train_pred = data.get("y_train_pred_transformed") if hasattr(data, "get") else None
    # np.load doesn't expose .get; use membership check
    y_train_true = data["y_train_true_transformed"] if "y_train_true_transformed" in data.files else None
    y_train_pred = data["y_train_pred_transformed"] if "y_train_pred_transformed" in data.files else None

    run_dir = pred_path.parent
    name = run_dir.name
    parity_plot(
        y_true, y_pred,
        y_train_true=y_train_true, y_train_pred=y_train_pred,
        title=f"{name}",
        xlabel="Actual",
        ylabel="Predicted",
        save_path=run_dir / "parity_plot.png",
    )
    logger.info("  %s: re-rendered parity_plot.png", run_dir)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()

    pred_files = sorted(args.results_dir.rglob("predictions.npz"))
    if not pred_files:
        logger.warning("No predictions.npz found under %s", args.results_dir)
        return 0

    logger.info("Re-rendering %d parity plot(s) ...", len(pred_files))
    for pf in pred_files:
        try:
            rerender_one(pf)
        except Exception as e:  # noqa: BLE001
            logger.error("  %s: failed — %s", pf, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
