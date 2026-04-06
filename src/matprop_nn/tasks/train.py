"""Training entry point — reads a YAML config and runs the full pipeline.

The ``model.arch`` field in the config selects which :class:`ModelEngine`
to use (``"tensornet"``, ``"chgnet"``, ``"alignn"``).  Everything else
(data loading, splitting, logging) is shared.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightning as L
import numpy as np
import yaml
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from pymatgen.core import Structure

from matprop_nn.models import get_engine, AVAILABLE_ARCHS
from matprop_nn.tasks.regression import RegressionModule

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Shared data loading ──────────────────────────────────────────────

def load_structures_and_targets(
    json_path: str | Path,
    target_key: str,
) -> tuple[list[Structure], list[float], list[str]]:
    """Read JSON, parse pymatgen structures and scalar targets."""
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    structures, targets, material_ids = [], [], []
    skipped = 0
    for rec in records:
        struct_dict = rec.get("structure")
        target_val = rec.get(target_key)
        if struct_dict is None or target_val is None:
            skipped += 1
            continue
        try:
            structures.append(Structure.from_dict(struct_dict))
        except Exception:
            skipped += 1
            continue
        targets.append(float(target_val))
        material_ids.append(str(rec.get("material_id", "")))

    if skipped:
        logger.warning("Skipped %d records (missing structure or target).", skipped)
    logger.info("Loaded %d structures with target '%s'.", len(structures), target_key)
    return structures, targets, material_ids


# ── Main run ──────────────────────────────────────────────────────────

def run(cfg: dict) -> None:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    arch = model_cfg.get("arch", "tensornet")
    logger.info("Architecture: %s", arch)
    engine = get_engine(arch)

    # 1. Load raw data (shared across all architectures)
    structures, targets, material_ids = load_structures_and_targets(
        data_cfg["json_path"], data_cfg["target_key"],
    )
    targets_np = np.array(targets)
    data_mean, data_std = float(targets_np.mean()), float(targets_np.std())
    logger.info("Target stats — mean: %.4f  std: %.4f", data_mean, data_std)

    # 2. Split (shared indices — fair comparison)
    indices = list(range(len(structures)))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=train_cfg.get("test_frac", 0.1),
        random_state=train_cfg.get("seed", 42),
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=train_cfg.get("val_frac", 0.1)
        / (1 - train_cfg.get("test_frac", 0.1)),
        random_state=train_cfg.get("seed", 42),
    )

    # 3. Engine-specific: datasets → dataloaders → model
    from matgl.ext._pymatgen_pyg import get_element_list

    element_types = get_element_list(structures)

    train_ds, val_ds, test_ds = engine.prepare_datasets(
        structures, targets, train_idx, val_idx, test_idx,
        cutoff=model_cfg.get("cutoff", 5.0),
        element_types=element_types,
        cache_dir=data_cfg.get("cache_dir", "MGLDataset"),
        target_key=data_cfg.get("target_key", "target"),
        **{k: v for k, v in model_cfg.items() if k not in ("arch", "cutoff")},
    )

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)
    train_loader, val_loader, test_loader = engine.build_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=batch_size, num_workers=num_workers,
    )

    model = engine.build_model(element_types=element_types, **model_cfg)

    # 4. Lightning module (shared)
    lit_module = RegressionModule(
        model,
        engine,
        data_mean=data_mean,
        data_std=data_std,
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    log_dir = train_cfg.get("log_dir", "logs")
    csv_logger = CSVLogger(log_dir, name=f"regression_{arch}")

    trainer = L.Trainer(
        max_epochs=train_cfg.get("max_epochs", 100),
        accelerator=train_cfg.get("accelerator", "auto"),
        logger=csv_logger,
        default_root_dir=log_dir,
    )
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    test_results = trainer.test(lit_module, dataloaders=test_loader)
    logger.info("Test results: %s", test_results)


def main():
    parser = argparse.ArgumentParser(description="Train a regression model on material data.")
    parser.add_argument("config", type=Path, help="Path to YAML config file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
