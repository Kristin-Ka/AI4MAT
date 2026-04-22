"""Matminer-based feature generation pipeline.

Two feature groups:
  1. **Composition features** (~130 dims): elemental statistics, stoichiometry,
     valence orbital, ionic properties — no 3D information.
  2. **Density / lattice features** (~3 dims): density, volume-per-atom,
     packing fraction — weak 3D (uses lattice parameters but not atomic coords).

The pipeline is fit/transform compatible: fit on training data, transform on
val/test to avoid data leakage in imputation and scaling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _get_composition_featurizers():
    """Return matminer composition featurizers (no 3D info)."""
    from matminer.featurizers.composition import (
        ElementProperty,
        Stoichiometry,
        ValenceOrbital,
        IonProperty,
    )
    return [
        ("elem_prop", ElementProperty.from_preset("magpie")),
        ("stoich", Stoichiometry()),
        ("valence", ValenceOrbital()),
        ("ion", IonProperty()),
    ]


def _get_density_featurizers():
    """Return matminer structure-level featurizers (weak 3D — lattice only)."""
    from matminer.featurizers.structure import DensityFeatures
    return [
        ("density", DensityFeatures()),
    ]


class MatminerFeaturizer:
    """Generate and cache matminer feature matrices.

    Parameters
    ----------
    feature_groups : list of str
        Which feature groups to include: ``"composition"``, ``"density"``.
    n_jobs : int
        Parallelism for matminer featurize calls.
    """

    def __init__(
        self,
        feature_groups: list[str] | None = None,
        n_jobs: int = 1,
    ):
        if feature_groups is None:
            feature_groups = ["composition", "density"]
        self.feature_groups = feature_groups
        self.n_jobs = n_jobs
        self._featurizers: list[tuple[str, Any]] = []
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] = []

    def _init_featurizers(self):
        self._featurizers = []
        if "composition" in self.feature_groups:
            self._featurizers.extend(_get_composition_featurizers())
        if "density" in self.feature_groups:
            self._featurizers.extend(_get_density_featurizers())

    def featurize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all featurizers to a DataFrame that has ``composition`` and
        optionally ``structure`` columns.

        Returns a DataFrame of numeric features only.
        """
        from pymatgen.core import Composition

        self._init_featurizers()

        if "composition" not in df.columns and "formula" in df.columns:
            df = df.copy()
            df["composition"] = df["formula"].apply(Composition)
        elif "composition" not in df.columns and "formula_pretty" in df.columns:
            df = df.copy()
            df["composition"] = df["formula_pretty"].apply(Composition)

        feat_frames = []
        for name, featurizer in self._featurizers:
            featurizer.set_n_jobs(self.n_jobs)
            try:
                if name == "density":
                    if "structure" not in df.columns:
                        logger.warning(
                            "Skipping density features: no 'structure' column."
                        )
                        continue
                    result = featurizer.featurize_dataframe(
                        df, col_id="structure", ignore_errors=True,
                    )
                else:
                    result = featurizer.featurize_dataframe(
                        df, col_id="composition", ignore_errors=True,
                    )
                labels = featurizer.feature_labels()
                feat_frames.append(result[labels])
                logger.info("  %s: %d features", name, len(labels))
            except Exception:
                logger.warning("Featurizer '%s' failed, skipping.", name, exc_info=True)

        if not feat_frames:
            raise RuntimeError("All featurizers failed — no features generated.")

        features = pd.concat(feat_frames, axis=1)
        features = features.apply(pd.to_numeric, errors="coerce")
        self._feature_names = list(features.columns)
        logger.info("Total raw features: %d", len(self._feature_names))
        return features

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit imputer + scaler on training data and return transformed array."""
        self._pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        return self._pipeline.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using the already-fit pipeline."""
        if self._pipeline is None:
            raise RuntimeError("Call fit_transform first.")
        return self._pipeline.transform(X)

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def save_features(
        self,
        X: np.ndarray,
        material_ids: list[str],
        path: str | Path,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(X, columns=self._feature_names)
        df.insert(0, "material_id", material_ids)
        df.to_parquet(path, index=False)
        logger.info("Saved features (%s) to %s", df.shape, path)
        return path

    @staticmethod
    def load_features(path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(path)
