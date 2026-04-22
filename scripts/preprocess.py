#!/usr/bin/env python3
"""Preprocess raw MP JSON into task-specific datasets and a unified split file.

Steps:
  1. Load the raw JSON dump (from fetch_mp_dataset.py)
  2. Filter records per task (only entries with valid target)
  3. Report dataset sizes per property
  4. Create and save a unified train/val/test split
  5. Optionally generate matminer features for descriptor models

Usage:
    python scripts/preprocess.py --json mp_materials.json --out-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_KEYS = {
    "e_total": "total_dielectric",
    "e_ionic": "ionic_dielectric",
    "band_gap": "band_gap",
    "energy_above_hull": "energy_above_hull",
}

METADATA_FIELDS = [
    "material_id",
    "formula_pretty",
    "nsites",
    "nelements",
    "volume",
    "density",
]


def load_records(json_path: str | Path) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def filter_for_task(records: list[dict], target_key: str) -> list[dict]:
    """Keep only records that have both a valid structure and the target."""
    valid = []
    for rec in records:
        if rec.get("structure") is None:
            continue
        val = rec.get(target_key)
        if val is None:
            continue
        try:
            float(val)
        except (TypeError, ValueError):
            continue
        valid.append(rec)
    return valid


def summarise(records: list[dict], target_key: str) -> dict:
    vals = np.array([float(r[target_key]) for r in records])
    return {
        "count": len(vals),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "median": float(np.median(vals)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess MP data for benchmark.")
    parser.add_argument("--json", type=Path, required=True, help="Raw MP JSON file.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--generate-features", action="store_true",
                        help="Also run matminer featurization (slow).")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data from %s ...", args.json)
    records = load_records(args.json)
    logger.info("Total raw records: %d", len(records))

    dataset_info = {}
    for raw_key, friendly_name in TARGET_KEYS.items():
        valid = filter_for_task(records, raw_key)
        stats = summarise(valid, raw_key)
        dataset_info[friendly_name] = {"raw_key": raw_key, **stats}
        logger.info(
            "  %-25s  N=%d  mean=%.4f  std=%.4f  [%.4f, %.4f]",
            friendly_name, stats["count"], stats["mean"], stats["std"],
            stats["min"], stats["max"],
        )

        mids = [r["material_id"] for r in valid]
        from matprop_nn.utils.splits import create_splits, save_splits
        split = create_splits(
            mids,
            test_frac=args.test_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        split_path = args.split_dir / f"split_{friendly_name}.json"
        save_splits(split, split_path)

    with open(args.out_dir / "dataset_summary.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    logger.info("Dataset summary saved to %s", args.out_dir / "dataset_summary.json")

    if args.generate_features:
        logger.info("Generating matminer features (this may take a while)...")
        from pymatgen.core import Structure, Composition
        from matprop_nn.features.matminer_features import MatminerFeaturizer

        all_valid = [r for r in records if r.get("structure") is not None]
        logger.info("Featurizing %d structures...", len(all_valid))

        structs = []
        mids = []
        formulas = []
        compositions = []
        for rec in all_valid:
            try:
                s = Structure.from_dict(rec["structure"])
                structs.append(s)
                mids.append(rec["material_id"])
                formulas.append(rec.get("formula_pretty", s.composition.reduced_formula))
                compositions.append(s.composition)
            except Exception:
                continue

        df = pd.DataFrame({
            "material_id": mids,
            "formula_pretty": formulas,
            "composition": compositions,
            "structure": structs,
        })

        featurizer = MatminerFeaturizer(
            feature_groups=["composition", "density"],
            n_jobs=args.n_jobs,
            cache_dir=args.out_dir / "_feat_cache",
        )
        feat_df = featurizer.featurize_df(df, material_ids=mids)
        featurizer.save_features(feat_df.values, mids, args.out_dir / "matminer_features.parquet")

    logger.info("Preprocessing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
