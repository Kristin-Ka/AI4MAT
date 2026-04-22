#!/usr/bin/env python3
"""Evaluate a trained model checkpoint on the test set and produce plots.

Usage:
    python scripts/evaluate.py configs/default.yaml path/to/best.ckpt --out-dir results/eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightning as L
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from matprop_nn.evaluation.metrics import compute_metrics
from matprop_nn.evaluation.plots import parity_plot, error_histogram
from matprop_nn.models import get_engine
from matprop_nn.tasks.train import load_structures_and_targets
from matprop_nn.tasks.regression import RegressionModule
from matprop_nn.utils.splits import load_splits, ids_to_indices

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on test split.")
    parser.add_argument("config", type=Path, help="YAML config (same as training).")
    parser.add_argument("checkpoint", type=Path, help="Lightning .ckpt file.")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    arch = model_cfg.get("arch", "tensornet")
    target_key = data_cfg["target_key"]
    engine = get_engine(arch)

    structures, targets, material_ids = load_structures_and_targets(
        data_cfg["json_path"], target_key,
    )
    targets_np = np.array(targets)
    data_mean, data_std = float(targets_np.mean()), float(targets_np.std())

    split_file = data_cfg.get("split_file")
    if split_file and Path(split_file).exists():
        split = load_splits(split_file)
        train_idx, val_idx, test_idx = ids_to_indices(material_ids, split)
    else:
        indices = list(range(len(structures)))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=train_cfg.get("test_frac", 0.1),
            random_state=train_cfg.get("seed", 42),
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=train_cfg.get("val_frac", 0.1) / (1 - train_cfg.get("test_frac", 0.1)),
            random_state=train_cfg.get("seed", 42),
        )

    from matgl.ext._pymatgen_pyg import get_element_list
    element_types = get_element_list(structures)

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
    trainer.test(lit_module, dataloaders=test_loader)

    y_pred, y_true = lit_module.get_test_predictions()
    y_pred_np, y_true_np = y_pred.numpy(), y_true.numpy()
    metrics = compute_metrics(y_true_np, y_pred_np)

    out_dir = args.out_dir or Path(f"results/eval_{arch}_{target_key}")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "predictions.npz", y_true=y_true_np, y_pred=y_pred_np)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    parity_plot(
        y_true_np, y_pred_np, title=f"{arch} — {target_key}",
        metrics=metrics, save_path=out_dir / "parity_plot.png",
    )
    error_histogram(
        y_true_np, y_pred_np, title=f"{arch} — {target_key}",
        save_path=out_dir / "error_histogram.png",
    )

    print(f"Test metrics: {json.dumps(metrics, indent=2)}")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
