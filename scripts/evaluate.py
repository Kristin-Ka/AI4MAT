#!/usr/bin/env python3
"""Evaluate a trained model checkpoint on the test set."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightning as L
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from matprop_nn.models import get_engine
from matprop_nn.tasks.train import load_structures_and_targets
from matprop_nn.tasks.regression import RegressionModule
from matgl.ext._pymatgen_pyg import get_element_list

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on test split.")
    parser.add_argument("config", type=Path, help="YAML config (same as training).")
    parser.add_argument("checkpoint", type=Path, help="Lightning .ckpt file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    arch = model_cfg.get("arch", "tensornet")
    engine = get_engine(arch)

    structures, targets, _ = load_structures_and_targets(
        data_cfg["json_path"], data_cfg["target_key"],
    )
    targets_np = np.array(targets)
    data_mean, data_std = float(targets_np.mean()), float(targets_np.std())

    element_types = get_element_list(structures)

    indices = list(range(len(structures)))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=train_cfg.get("test_frac", 0.1),
        random_state=train_cfg.get("seed", 42),
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=train_cfg.get("val_frac", 0.1) / (1 - train_cfg.get("test_frac", 0.1)),
        random_state=train_cfg.get("seed", 42),
    )

    _, _, test_ds = engine.prepare_datasets(
        structures, targets, train_idx, val_idx, test_idx,
        element_types=element_types,
        cache_dir=data_cfg.get("cache_dir", "MGLDataset"),
        target_key=data_cfg.get("target_key", "target"),
        **{k: v for k, v in model_cfg.items() if k not in ("arch",)},
    )

    _, _, test_loader = engine.build_dataloaders(
        test_ds, test_ds, test_ds,
        batch_size=train_cfg.get("batch_size", 32),
    )

    model = engine.build_model(element_types=element_types, **model_cfg)

    lit_module = RegressionModule.load_from_checkpoint(
        args.checkpoint,
        model=model,
        engine=engine,
        data_mean=data_mean,
        data_std=data_std,
    )

    trainer = L.Trainer(accelerator=train_cfg.get("accelerator", "auto"), logger=False)
    results = trainer.test(lit_module, dataloaders=test_loader)
    print("Test results:", results)


if __name__ == "__main__":
    main()
