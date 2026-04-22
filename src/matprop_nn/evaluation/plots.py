"""Plotting utilities for benchmark evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _compute_stats(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return (R², RMSE) for a single split."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return r2, rmse


def parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_train_true: np.ndarray | None = None,
    y_train_pred: np.ndarray | None = None,
    title: str = "",
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    metrics: dict[str, float] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (6.5, 6.5),
) -> plt.Figure:
    """Parity plot with train/test overlay and marginal distributions.

    Rendered as a joint plot with a central scatter (train in grey, test in
    blue), marginal KDE-style histograms along the top (actual values) and
    right (predicted values), an identity y=x line, and an annotation box
    showing N, train R²/RMSE, test R²/RMSE.  If ``y_train_*`` is ``None``,
    only the test split is drawn (falls back to the prior behaviour).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    has_train = y_train_true is not None and y_train_pred is not None
    if has_train:
        y_train_true = np.asarray(y_train_true, dtype=float)
        y_train_pred = np.asarray(y_train_pred, dtype=float)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2, width_ratios=(5, 1), height_ratios=(1, 5),
        left=0.12, right=0.95, bottom=0.1, top=0.93,
        wspace=0.04, hspace=0.04,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    train_color = "#4a4a4a"
    test_color = "#2c5fcf"

    if has_train:
        ax.scatter(
            y_train_true, y_train_pred,
            s=6, alpha=0.35, c=train_color, edgecolors="none",
            label="Train", zorder=2,
        )
    ax.scatter(
        y_true, y_pred,
        s=12, alpha=0.75, c=test_color, edgecolors="none",
        label="Test", zorder=3,
    )

    # Identity line spans the union of all data.
    if has_train:
        all_lo = min(y_true.min(), y_pred.min(), y_train_true.min(), y_train_pred.min())
        all_hi = max(y_true.max(), y_pred.max(), y_train_true.max(), y_train_pred.max())
    else:
        all_lo = min(y_true.min(), y_pred.min())
        all_hi = max(y_true.max(), y_pred.max())
    margin = (all_hi - all_lo) * 0.05 if all_hi > all_lo else 1.0
    lo, hi = all_lo - margin, all_hi + margin
    ax.plot([lo, hi], [lo, hi], color="k", lw=1, zorder=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9, markerscale=2)

    # Marginal distributions — smooth KDE curves to match the reference
    # figure style.  Falls back to a histogram if the sample has near-zero
    # variance (which breaks Gaussian KDE).
    xs = np.linspace(lo, hi, 400)

    def _kde_or_hist(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 2 or v.std() < 1e-9:
            return None
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(v, bw_method="scott")
            return xs, kde(xs)
        except Exception:
            return None

    def _draw_kde(axis, values, color, alpha_fill, orientation):
        curve = _kde_or_hist(values)
        if curve is None:
            bins = np.linspace(lo, hi, 50)
            if orientation == "horizontal":
                axis.hist(values, bins=bins, color=color, alpha=alpha_fill,
                          density=True, edgecolor="none",
                          orientation="horizontal")
            else:
                axis.hist(values, bins=bins, color=color, alpha=alpha_fill,
                          density=True, edgecolor="none")
            return
        x_curve, y_curve = curve
        if orientation == "horizontal":
            axis.fill_betweenx(x_curve, 0, y_curve, color=color,
                               alpha=alpha_fill, linewidth=0)
            axis.plot(y_curve, x_curve, color=color, lw=1.2, alpha=0.9)
        else:
            axis.fill_between(x_curve, 0, y_curve, color=color,
                              alpha=alpha_fill, linewidth=0)
            axis.plot(x_curve, y_curve, color=color, lw=1.2, alpha=0.9)

    if has_train:
        _draw_kde(ax_top, y_train_true, train_color, 0.35, "vertical")
        _draw_kde(ax_right, y_train_pred, train_color, 0.35, "horizontal")
    _draw_kde(ax_top, y_true, test_color, 0.35, "vertical")
    _draw_kde(ax_right, y_pred, test_color, 0.35, "horizontal")

    for spine in ("top", "right", "left"):
        ax_top.spines[spine].set_visible(False)
    for spine in ("top", "right", "bottom"):
        ax_right.spines[spine].set_visible(False)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", left=False, labelleft=False)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(axis="x", bottom=False, labelbottom=False)

    # Stats annotation — follows the reference layout:
    #   N = <n_train or n_total>
    #   Train R² / Train RMSE
    #   Test R² / Test RMSE
    annotation_lines: list[str] = []
    if has_train:
        r2_tr, rmse_tr = _compute_stats(y_train_true, y_train_pred)
        r2_te, rmse_te = _compute_stats(y_true, y_pred)
        annotation_lines.append(f"N = {len(y_train_true) + len(y_true)}")
        annotation_lines.append(f"Train R² = {r2_tr:.2f}")
        annotation_lines.append(f"Train RMSE = {rmse_tr:.2f}")
        annotation_lines.append(f"Test R² = {r2_te:.2f}")
        annotation_lines.append(f"Test RMSE = {rmse_te:.2f}")
    else:
        r2_te, rmse_te = _compute_stats(y_true, y_pred)
        annotation_lines.append(f"N = {len(y_true)}")
        annotation_lines.append(f"Test R² = {r2_te:.2f}")
        annotation_lines.append(f"Test RMSE = {rmse_te:.2f}")
    if metrics:
        for k, v in metrics.items():
            if k.upper() not in {"RMSE", "R2", "R²", "MAE"}:
                annotation_lines.append(f"{k}: {v:.3f}")

    ax.text(
        0.97, 0.04, "\n".join(annotation_lines),
        transform=ax.transAxes, fontsize=9,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#aaaaaa", alpha=0.9),
    )

    if title:
        fig.suptitle(title, fontsize=11, y=0.995)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "",
    save_path: str | Path | None = None,
    bins: int = 50,
) -> plt.Figure:
    """Histogram of prediction errors (pred - actual)."""
    errors = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(errors, bins=bins, edgecolor="white", color="#3366cc", alpha=0.8)
    ax.axvline(0, color="k", linestyle="--", lw=0.8)
    ax.set_xlabel("Error (predicted − actual)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def comparison_bar_chart(
    results: dict[str, dict[str, float]],
    metric: str = "RMSE",
    *,
    title: str = "",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Bar chart comparing multiple models on a single metric.

    Parameters
    ----------
    results : {model_name: {metric_name: value, ...}}
    """
    models = list(results.keys())
    values = [results[m].get(metric, 0.0) for m in models]

    if figsize is None:
        figsize = (max(5, len(models) * 1.0), 4)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, values, color=colors, edgecolor="white")

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{v:.3f}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel(metric)
    ax.set_title(title or f"Model comparison — {metric}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def multi_property_comparison(
    all_results: dict[str, dict[str, dict[str, float]]],
    metric: str = "RMSE",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart: properties on x-axis, models as groups.

    Parameters
    ----------
    all_results : {property: {model: {metric: value}}}
    """
    properties = list(all_results.keys())
    models = sorted({m for prop in all_results.values() for m in prop})
    n_props = len(properties)
    n_models = len(models)

    x = np.arange(n_props)
    width = 0.8 / n_models
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(max(7, n_props * 2), 5))
    for i, model in enumerate(models):
        vals = [all_results[p].get(model, {}).get(metric, 0.0) for p in properties]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model, color=colors[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(properties, rotation=15, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Benchmark — {metric}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
