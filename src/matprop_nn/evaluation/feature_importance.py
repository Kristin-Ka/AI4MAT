"""Feature-importance analysis and visualisation for tree-based models.

For gradient-boosted trees (XGBoost, LightGBM) there are three commonly used
importance scores per feature:

* ``gain``   — total gain across every split using the feature (how much
  the feature reduces loss overall).  **Default and most informative.**
* ``weight`` — number of splits the feature appears in.  Biased toward
  high-cardinality features.
* ``cover``  — average number of training samples affected by splits on
  the feature.

This module extracts all three, writes a tidy CSV, and renders a
horizontal bar plot of the top-N features ranked by gain.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_xgb_feature_importance(
    xgb_model,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return a DataFrame of per-feature gain / weight / cover importances.

    Columns: ``feature``, ``gain``, ``weight``, ``cover``, sorted by gain
    descending.  Works on ``xgboost.XGBRegressor``,
    ``xgboost.XGBClassifier`` or any object exposing ``.get_booster()``.
    """
    booster = xgb_model.get_booster()

    importance_types = ["gain", "weight", "cover"]
    scores = {}
    for imp_type in importance_types:
        # booster.get_score uses internal ``f{idx}`` names when trained on
        # a plain numpy array; build a full mapping so features absent from
        # the tree ensemble get a zero score.
        raw = booster.get_score(importance_type=imp_type)
        scores[imp_type] = np.zeros(len(feature_names), dtype=float)
        for k, v in raw.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if 0 <= idx < len(feature_names):
                    scores[imp_type][idx] = v
            elif k in feature_names:
                scores[imp_type][feature_names.index(k)] = v

    df = pd.DataFrame({
        "feature": feature_names,
        "gain": scores["gain"],
        "weight": scores["weight"],
        "cover": scores["cover"],
    })
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)
    total = df["gain"].sum()
    df["gain_normalized"] = df["gain"] / total if total > 0 else 0.0
    return df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_k: int = 30,
    title: str = "Feature Importance (XGBoost gain)",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> Path | None:
    """Render a horizontal bar plot of the top-K features ranked by gain."""
    import matplotlib.pyplot as plt

    top = importance_df.head(top_k).iloc[::-1]  # flip so highest-gain on top

    if figsize is None:
        figsize = (8, max(4, 0.3 * len(top) + 1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top["feature"], top["gain"], color="#3b7dd8", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Gain (cumulative loss reduction)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved feature-importance plot to %s", save_path)
        plt.close(fig)
        return save_path

    plt.close(fig)
    return None


def save_feature_importance(
    importance_df: pd.DataFrame,
    out_dir: str | Path,
    prefix: str = "feature_importance",
    top_k: int = 30,
    title: str = "Feature Importance",
) -> dict[str, Path]:
    """Save the full importance table (CSV) and a top-K bar plot (PNG).

    Returns a ``{"csv": ..., "plot": ...}`` dict of output paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}.csv"
    plot_path = out_dir / f"{prefix}_top{top_k}.png"

    importance_df.to_csv(csv_path, index=False)
    logger.info(
        "Saved feature importance table (%d features) to %s",
        len(importance_df), csv_path,
    )
    plot_feature_importance(
        importance_df, top_k=top_k, title=title, save_path=plot_path,
    )
    return {"csv": csv_path, "plot": plot_path}
