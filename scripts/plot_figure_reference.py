"""Reproduce the reference figure from the Schrödinger slide.

Left panel  : Test RMSE on dielectric tasks (Total Dielectric, Ionic
              Dielectric) — shared y-axis.
Right panel : Test RMSE on Band Gap + Energy Above Hull — dual y-axis
              because the two targets have very different magnitudes
              (~0.5 eV vs ~0.1 eV).

Four bars per group: MatMiner, MatMiner3D, M3GNet, ALIGNN.

Reads aggregated metrics from ``results/REPORT.md`` sources directly
(``results/<model>/<arch>_<task>/metrics.json``) so any missing run
simply shows up as a blank bar rather than crashing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# Map human labels to (results/<dir>/<arch>_<task>/metrics.json) lookups.
# The ``dir`` is the top-level folder under ``results/`` (what
# run_benchmark.py passes to ``--log_dir``), the ``arch`` is the value
# in the model config (used by train.py to build the run subfolder).
MODEL_SPECS = [
    ("MatMiner",   "xgboost",     "xgboost",  "#1f77b4"),
    ("MatMiner3D", "xgboost_3d",  "xgboost",  "#ff7f0e"),
    ("M3GNet",     "m3gnet",      "m3gnet",   "#2ca02c"),
    ("ALIGNN",     "alignn",      "alignn",   "#7f7f7f"),
]

# (task_key, display_name, y_label_suffix)
DIELECTRIC_TASKS = [
    ("e_total", "Total Dielectric"),
    ("e_ionic", "Ionic Dielectric"),
]
GAP_HULL_TASKS = [
    ("band_gap",          "Bandgap",             "Bandgap"),
    ("energy_above_hull", "Energy\nAbove Hull",  "Energy Above Hull"),
]


def _load_rmse(results_dir: Path, model_dir: str, arch: str, task: str) -> float | None:
    """Return RMSE in *original* units for a given model / task; None if missing."""
    metrics_path = results_dir / model_dir / f"{arch}_{task}" / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            m = json.load(f)
    except Exception:
        return None
    # Metrics reported in original units when a log1p transform is active,
    # else the top-level RMSE is already in original units.
    orig = m.get("original_units")
    if isinstance(orig, dict) and "RMSE" in orig:
        return float(orig["RMSE"])
    return float(m.get("RMSE", 0.0))


def _draw_bars(
    ax: plt.Axes,
    tasks: list[tuple],
    rmses: dict[str, list[float | None]],
    *,
    ylabel: str,
    annotate: bool = True,
) -> None:
    """Draw grouped bars on ``ax``.

    ``tasks``  - list of (task_key, display_name) pairs driving the x-axis.
    ``rmses``  - {model_display_name: [rmse_per_task ...]}  (None → skipped).
    """
    n_tasks = len(tasks)
    n_models = len(MODEL_SPECS)
    x = np.arange(n_tasks)
    width = 0.8 / n_models

    for i, (label, _, _, color) in enumerate(MODEL_SPECS):
        vals = rmses.get(label, [None] * n_tasks)
        offset = (i - n_models / 2 + 0.5) * width
        heights = [0.0 if v is None else v for v in vals]
        bars = ax.bar(
            x + offset, heights, width,
            label=label, color=color, edgecolor="black", linewidth=0.4,
        )
        if annotate:
            for bar, v in zip(bars, vals):
                if v is None:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.2f}" if v >= 0.1 else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([display for _, display, *_ in tasks])
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_figure(results_dir: Path, save_path: Path) -> None:
    diel_rmses: dict[str, list[float | None]] = {}
    for label, model_dir, arch, _ in MODEL_SPECS:
        diel_rmses[label] = [
            _load_rmse(results_dir, model_dir, arch, t) for t, _ in DIELECTRIC_TASKS
        ]

    gap_rmse = {label: _load_rmse(results_dir, md, ar, "band_gap")
                for label, md, ar, _ in MODEL_SPECS}
    hull_rmse = {label: _load_rmse(results_dir, md, ar, "energy_above_hull")
                 for label, md, ar, _ in MODEL_SPECS}

    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.25)

    ax_left = fig.add_subplot(gs[0, 0])
    _draw_bars(ax_left, DIELECTRIC_TASKS, diel_rmses, ylabel="Test RMSE")
    ax_left.set_title(
        "Stoichiometry and Graph-Based ML Models\n"
        "MatMiner Featurizers (w/ and w/o 3D information) vs M3GNet and ALIGNN "
        "(3D information)",
        fontsize=10, loc="left", pad=12,
    )

    ax_right = fig.add_subplot(gs[0, 1])
    ax_right2 = ax_right.twinx()

    n_models = len(MODEL_SPECS)
    width = 0.8 / n_models
    x_gap = 0
    x_hull = 1

    for i, (label, _, _, color) in enumerate(MODEL_SPECS):
        offset = (i - n_models / 2 + 0.5) * width
        # Band gap on left axis
        v_gap = gap_rmse.get(label)
        if v_gap is not None:
            bar = ax_right.bar(
                x_gap + offset, v_gap, width, color=color,
                edgecolor="black", linewidth=0.4, label=label,
            )
            ax_right.text(
                x_gap + offset, v_gap * 1.01, f"{v_gap:.2f}",
                ha="center", va="bottom", fontsize=7,
            )
        # Energy above hull on right axis
        v_hull = hull_rmse.get(label)
        if v_hull is not None:
            ax_right2.bar(
                x_hull + offset, v_hull, width, color=color,
                edgecolor="black", linewidth=0.4,
            )
            ax_right2.text(
                x_hull + offset, v_hull * 1.01, f"{v_hull:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax_right.set_xticks([x_gap, x_hull])
    ax_right.set_xticklabels([t[1] for t in GAP_HULL_TASKS])
    ax_right.set_ylabel("Test RMSE (Bandgap)")
    ax_right2.set_ylabel("Test RMSE (Energy Above Hull)")
    # Hide the right-axis x-line-through artifact & give it right-side spine color
    ax_right.spines["top"].set_visible(False)
    ax_right2.spines["top"].set_visible(False)
    ax_right.legend(loc="upper right", fontsize=8, frameon=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", save_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/figure_reference.png"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    plot_figure(args.results_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
