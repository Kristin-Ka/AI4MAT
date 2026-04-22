"""Matminer-based feature generation pipeline.

Two feature groups:
  1. **Composition features** (~130 dims): elemental statistics, stoichiometry,
     valence orbital, ionic properties — no 3D information.
  2. **Density / lattice features** (~3 dims): density, volume-per-atom,
     packing fraction — weak 3D (uses lattice parameters but not atomic coords).

Per-featurizer checkpointing: each featurizer's output is written to
``<cache_dir>/<name>.parquet`` as soon as it finishes.  On re-run, cached
outputs are loaded from disk and the featurizer is skipped.  This makes the
pipeline robust to multiprocessing hangs (e.g. CPython #105826, which can
cause matminer's ``Pool`` to hang on cleanup after IonProperty on 24-core
hosts) — the user can Ctrl-C, re-run, and pick up where it stopped.

The pipeline is fit/transform compatible: fit on training data, transform on
val/test to avoid data leakage in imputation and scaling.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _get_composition_featurizers():
    """Return matminer composition featurizers (no 3D info).

    ``IonProperty`` uses ``fast=True`` to skip the full oxidation-state
    enumeration (``Composition.oxi_state_guesses``), which is O(n!) in the
    number of elements and can take tens of seconds per structure on complex
    compositions.  The ``fast`` path assumes each element takes a single
    oxidation state; it loses accuracy on heterovalent compounds (e.g.
    Fe3O4) but featurizes ~25× faster with only a minor impact on the three
    ionic-character features it produces.
    """
    from matminer.featurizers.composition import (
        ElementProperty,
        IonProperty,
        Stoichiometry,
        ValenceOrbital,
    )
    return [
        ("elem_prop", ElementProperty.from_preset("magpie")),
        ("stoich", Stoichiometry()),
        ("valence", ValenceOrbital()),
        ("ion", IonProperty(fast=True)),
    ]


def _get_density_featurizers():
    """Return matminer structure-level featurizers (weak 3D — lattice only)."""
    from matminer.featurizers.structure import DensityFeatures
    return [
        ("density", DensityFeatures()),
    ]


def _get_structure3d_featurizers():
    """Return matminer 3D-structure featurizers (use atomic coordinates).

    This matches the "MatMiner3D" baseline in the Schrödinger / Petousis
    2017 / Dunn 2020 (Matbench) papers, which augment the composition /
    density feature set with structure-level descriptors derived from
    the actual 3D atomic positions:

    * ``GlobalSymmetryFeatures`` — crystal system, space group number,
      centrosymmetry flag (5 features).
    * ``MaximumPackingEfficiency`` — radius-based packing fraction
      (1 feature).
    * ``StructuralHeterogeneity`` — mean, minimum relative difference of
      nearest-neighbour bond lengths; variance in coordination number
      (9 features).
    * ``ChemicalOrdering`` — Warren-Cowley short-range order parameters
      (3 features).  **Heavy**: ~4-6 GB RSS and very slow on Python 3.11
      because matminer's ``Pool(maxtasksperchild=1)`` cleanup hangs for
      10+ min after a 7 k-row pass (CPython #105826).  To mitigate, this
      featurizer is pinned to ``n_jobs=1`` via ``_SERIAL_FEATURIZERS``
      (bypasses the pool entirely), and results are cached per-material
      so a Ctrl-C + re-run continues where it stopped.  Include it only
      on machines with ≥ 24 GB RAM when not running a GNN concurrently.

    ``SineCoulombMatrix`` and ``OrbitalFieldMatrix`` are deliberately
    omitted — they add hundreds of features but are ~10× slower and rarely
    improve R² on the Matbench tasks we target.
    """
    from matminer.featurizers.structure import (
        ChemicalOrdering,
        GlobalSymmetryFeatures,
        MaximumPackingEfficiency,
        StructuralHeterogeneity,
    )
    return [
        ("sym", GlobalSymmetryFeatures()),
        ("packing", MaximumPackingEfficiency()),
        ("heterogeneity", StructuralHeterogeneity()),
        ("ordering", ChemicalOrdering()),
    ]


class MatminerFeaturizer:
    """Generate and cache matminer feature matrices.

    Parameters
    ----------
    feature_groups : list of str
        Featurizers to include.  Accepts both **group aliases** and
        **individual featurizer names** (mix freely):

        * ``"composition"`` — all composition featurizers
          (``elem_prop + stoich + valence + ion``, 149 features)
        * ``"density"`` — structure density featurizer (3 features)
        * ``"structure3d"`` — 3D-structure featurizers (crystal symmetry +
          packing + bond heterogeneity + chemical ordering, ~18 features)
        * Individual names: ``"elem_prop"`` (132), ``"stoich"`` (6),
          ``"valence"`` (8), ``"ion"`` (3), ``"sym"``, ``"packing"``,
          ``"heterogeneity"``, ``"ordering"``.

        Example configurations:

        * ``["composition", "density"]`` — 152-feature MatMiner baseline
          (composition + weak-3D; matches Ward 2016 / Matbench)
        * ``["composition", "density", "structure3d"]`` — "MatMiner3D":
          ~170-feature set including true 3D descriptors (matches
          Petousis 2017 / Schrödinger benchmark)
        * ``["elem_prop", "density"]`` — 135-feature Magpie + density
    n_jobs : int
        Parallelism for matminer featurize calls.  Capped internally to
        avoid multiprocessing bugs.
    cache_dir : str | Path
        Directory for per-featurizer checkpoints.
    """

    # Cap multiprocessing workers to avoid a Python 3.11 race/hang in
    # ``multiprocessing.Pool`` where concurrent FD registration after many
    # Pool lifecycles raises ``KeyError: '<fd> is already registered'`` or
    # hangs Pool cleanup (CPython #105826).  8 workers gives near-max
    # throughput while staying below the observed race threshold.
    _N_JOBS_CAP = 8

    # Some matminer featurizers call ``multiprocessing.Pool.__exit__`` with
    # ``maxtasksperchild=1``, which on Python 3.11 can hang the main
    # process for 10+ minutes during Pool cleanup after tens of thousands
    # of worker spawn/kill cycles (observed on ChemicalOrdering at 7k
    # rows).  Force these featurizers to run serially — they're single-
    # CPU-thread internally anyway, so the wall-clock cost is similar to
    # the parallel version but without the risk of a Pool-cleanup hang.
    _SERIAL_FEATURIZERS = {"ordering"}

    # Canonical names for individual featurizers — used to validate
    # ``feature_groups`` and report unknown entries.
    _COMPOSITION_NAMES = {"elem_prop", "stoich", "valence", "ion"}
    _DENSITY_NAMES = {"density"}
    _STRUCTURE3D_NAMES = {"sym", "packing", "heterogeneity", "ordering"}
    _GROUP_ALIASES = {"composition", "density", "structure3d"}

    def __init__(
        self,
        feature_groups: list[str] | None = None,
        n_jobs: int = 1,
        cache_dir: str | Path = "data/processed/_feat_cache",
    ):
        if feature_groups is None:
            feature_groups = ["composition", "density"]
        self.feature_groups = feature_groups
        self.n_jobs = self._resolve_n_jobs(n_jobs)
        self.cache_dir = Path(cache_dir)
        self._featurizers: list[tuple[str, Any]] = []
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] = []

    @classmethod
    def _resolve_n_jobs(cls, n_jobs: int) -> int:
        """Matminer requires a positive int; -1 (sklearn) → cpu_count, capped."""
        import os
        if n_jobs is None or n_jobs == 0:
            return 1
        if n_jobs < 0:
            resolved = max(1, os.cpu_count() or 1)
        else:
            resolved = int(n_jobs)
        return min(resolved, cls._N_JOBS_CAP)

    def _init_featurizers(self):
        """Resolve ``feature_groups`` into a concrete list of featurizers.

        Group aliases expand to their members; individual names are kept
        as-is.  Duplicates (e.g. ``["composition", "elem_prop"]``) are
        silently de-duplicated while preserving first-appearance order.
        Unknown names raise ``ValueError`` early rather than silently
        producing an empty feature set.
        """
        requested: list[str] = []
        all_individual = (
            self._COMPOSITION_NAMES | self._DENSITY_NAMES | self._STRUCTURE3D_NAMES
        )
        for g in self.feature_groups:
            if g == "composition":
                requested.extend(["elem_prop", "stoich", "valence", "ion"])
            elif g == "density":
                requested.append("density")
            elif g == "structure3d":
                requested.extend(["sym", "packing", "heterogeneity", "ordering"])
            elif g in all_individual:
                requested.append(g)
            else:
                raise ValueError(
                    f"Unknown feature group '{g}'. Valid options: "
                    f"groups={sorted(self._GROUP_ALIASES)}, "
                    f"individual={sorted(all_individual)}"
                )

        seen, ordered = set(), []
        for name in requested:
            if name not in seen:
                seen.add(name)
                ordered.append(name)

        name_to_featurizer: dict[str, Any] = {}
        if any(n in self._COMPOSITION_NAMES for n in ordered):
            name_to_featurizer.update(dict(_get_composition_featurizers()))
        if any(n in self._DENSITY_NAMES for n in ordered):
            name_to_featurizer.update(dict(_get_density_featurizers()))
        if any(n in self._STRUCTURE3D_NAMES for n in ordered):
            name_to_featurizer.update(dict(_get_structure3d_featurizers()))

        self._featurizers = [(n, name_to_featurizer[n]) for n in ordered]
        logger.info(
            "Feature pipeline: %s", " + ".join(n for n, _ in self._featurizers),
        )

    # ------------------------------------------------------------------
    # Per-featurizer caching
    # ------------------------------------------------------------------
    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.parquet"

    def _load_cached(self, name: str, material_ids: list[str]) -> pd.DataFrame | None:
        path = self._cache_path(name)
        if not path.exists():
            return None
        try:
            cached = pd.read_parquet(path)
        except Exception:
            logger.warning("Could not read cache %s, regenerating.", path)
            return None
        if "material_id" not in cached.columns:
            return None
        id_to_row = dict(zip(cached["material_id"], cached.index))
        missing = [mid for mid in material_ids if mid not in id_to_row]
        if missing:
            logger.info(
                "Cache %s misses %d/%d IDs, regenerating.",
                name, len(missing), len(material_ids),
            )
            return None
        rows = [id_to_row[mid] for mid in material_ids]
        feat_cols = [c for c in cached.columns if c != "material_id"]
        out = cached.iloc[rows][feat_cols].reset_index(drop=True)
        logger.info("  %s: loaded %d features from cache (%s)", name, len(feat_cols), path)
        return out

    def _save_cached(self, name: str, feat_df: pd.DataFrame, material_ids: list[str]):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(name)
        out = feat_df.copy()
        out.insert(0, "material_id", material_ids)
        out.to_parquet(path, index=False)
        logger.info("  %s: cached to %s", name, path)

    # ------------------------------------------------------------------
    # Featurization
    # ------------------------------------------------------------------
    def featurize_df(
        self,
        df: pd.DataFrame,
        *,
        material_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Apply all featurizers to a DataFrame.

        The input DataFrame must contain a ``composition`` column (used by
        composition featurizers) and optionally a ``structure`` column (used
        by density featurizers).

        Per-featurizer outputs are cached to disk; cached outputs are reused
        on subsequent runs.  Pass ``material_ids`` (one per row) to match
        cached rows by ID.
        """
        from pymatgen.core import Composition

        self._init_featurizers()

        if "composition" not in df.columns and "formula" in df.columns:
            df = df.copy()
            df["composition"] = df["formula"].apply(Composition)
        elif "composition" not in df.columns and "formula_pretty" in df.columns:
            df = df.copy()
            df["composition"] = df["formula_pretty"].apply(Composition)

        if material_ids is None and "material_id" in df.columns:
            material_ids = df["material_id"].astype(str).tolist()
        if material_ids is None:
            material_ids = [str(i) for i in range(len(df))]

        feat_frames: list[pd.DataFrame] = []
        for name, featurizer in self._featurizers:
            cached = self._load_cached(name, material_ids)
            if cached is not None:
                feat_frames.append(cached)
                continue

            effective_n_jobs = 1 if name in self._SERIAL_FEATURIZERS else self.n_jobs
            featurizer.set_n_jobs(effective_n_jobs)
            col_id = (
                "structure"
                if (name in self._DENSITY_NAMES or name in self._STRUCTURE3D_NAMES)
                else "composition"
            )
            if col_id not in df.columns:
                logger.warning("Skipping %s: no '%s' column.", name, col_id)
                continue

            logger.info(
                "  %s: featurizing %d rows with n_jobs=%d ...",
                name, len(df), effective_n_jobs,
            )
            t0 = time.time()
            try:
                result = featurizer.featurize_dataframe(
                    df, col_id=col_id, ignore_errors=True,
                )
                labels = featurizer.feature_labels()
                sub = result[labels].reset_index(drop=True)
                feat_frames.append(sub)
                logger.info(
                    "  %s: done — %d features in %.1fs",
                    name, len(labels), time.time() - t0,
                )
                self._save_cached(name, sub, material_ids)
            except Exception:
                logger.warning(
                    "Featurizer '%s' failed, skipping.", name, exc_info=True,
                )

        if not feat_frames:
            raise RuntimeError("All featurizers failed — no features generated.")

        features = pd.concat(feat_frames, axis=1)
        features = features.apply(pd.to_numeric, errors="coerce")
        self._feature_names = list(features.columns)
        logger.info("Total raw features: %d", len(self._feature_names))
        return features

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------
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
