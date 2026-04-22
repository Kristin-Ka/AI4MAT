"""Plotting utilities for benchmark evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "",
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    metrics: dict[str, float] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (5, 5),
) -> plt.Figure:
    """Predicted-vs-actual scatter with identity line and marginal histograms."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, alpha=0.3, s=8, edgecolors="none", c="#3366cc")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k-", lw=1)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal")

    if metrics:
        text = "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        ax.text(
            0.05, 0.95, text, transform=ax.transAxes,
            fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
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
