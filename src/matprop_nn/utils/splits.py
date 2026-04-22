"""Reproducible train / val / test splitting utilities.

Splits are persisted as JSON so that every model uses exactly the same
partition. The file stores material IDs (not integer indices) to remain
stable even if the underlying JSON order changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split, KFold

logger = logging.getLogger(__name__)


def create_splits(
    material_ids: list[str],
    *,
    test_frac: float = 0.10,
    val_frac: float = 0.10,
    seed: int = 42,
    n_folds: int | None = None,
) -> dict:
    """Create a reproducible split and return a JSON-serialisable dict.

    Returns
    -------
    dict with keys ``train``, ``val``, ``test`` (lists of material_ids),
    plus metadata (``seed``, ``n_total``, etc.).
    If *n_folds* is given, adds a ``folds`` key with k-fold indices.
    """
    ids = np.array(material_ids)
    indices = np.arange(len(ids))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_frac, random_state=seed,
    )
    relative_val = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val, random_state=seed,
    )

    split = {
        "seed": seed,
        "test_frac": test_frac,
        "val_frac": val_frac,
        "n_total": len(ids),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train": ids[train_idx].tolist(),
        "val": ids[val_idx].tolist(),
        "test": ids[test_idx].tolist(),
    }

    if n_folds and n_folds > 1:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = []
        for fold_train, fold_val in kf.split(train_val_idx):
            folds.append({
                "train": ids[train_val_idx[fold_train]].tolist(),
                "val": ids[train_val_idx[fold_val]].tolist(),
            })
        split["folds"] = folds

    logger.info(
        "Split: %d train / %d val / %d test (seed=%d)",
        split["n_train"], split["n_val"], split["n_test"], seed,
    )
    return split


def save_splits(split: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
    logger.info("Saved split to %s", path)
    return path


def load_splits(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def ids_to_indices(material_ids: list[str], split: dict) -> tuple[list[int], list[int], list[int]]:
    """Convert material_id lists in *split* back to integer indices
    relative to the ordered *material_ids* list."""
    id_to_idx = {mid: i for i, mid in enumerate(material_ids)}

    def _lookup(id_list: list[str]) -> list[int]:
        return [id_to_idx[mid] for mid in id_list if mid in id_to_idx]

    return _lookup(split["train"]), _lookup(split["val"]), _lookup(split["test"])
