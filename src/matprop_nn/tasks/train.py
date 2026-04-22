"""Unified training entry point for both graph and descriptor models.

Reads a YAML config and orchestrates:
  1. Data loading and splitting
  2. Model-specific dataset preparation
  3. Training with early stopping and checkpointing
  4. Evaluation with metrics, predictions, and plots
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightning as L
import numpy as np
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from pymatgen.core import Structure

from matprop_nn.evaluation.metrics import compute_metrics
from matprop_nn.evaluation.plots import parity_plot, error_histogram
from matprop_nn.evaluation.benchmark import BenchmarkResult
from matprop_nn.models import get_engine, is_descriptor_model, AVAILABLE_ARCHS
from matprop_nn.tasks.regression import RegressionModule
from matprop_nn.utils.splits import create_splits, save_splits, load_splits, ids_to_indices

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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


def _get_or_create_split(
    material_ids: list[str],
    train_cfg: dict,
    data_cfg: dict,
) -> dict:
    """Load an existing split file or create a new one."""
    split_path = data_cfg.get("split_file")
    if split_path and Path(split_path).exists():
        logger.info("Loading existing split from %s", split_path)
        return load_splits(split_path)

    split = create_splits(
        material_ids,
        test_frac=train_cfg.get("test_frac", 0.10),
        val_frac=train_cfg.get("val_frac", 0.10),
        seed=train_cfg.get("seed", 42),
    )

    if split_path:
        save_splits(split, split_path)
    return split


def _run_graph_model(cfg: dict) -> BenchmarkResult | None:
    """Train and evaluate a graph-based model (TensorNet / CHGNet / ALIGNN / M3GNet)."""
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    arch = model_cfg.get("arch", "tensornet")
    target_key = data_cfg["target_key"]
    logger.info("Architecture: %s | Target: %s", arch, target_key)

    engine = get_engine(arch)

    structures, targets, material_ids = load_structures_and_targets(
        data_cfg["json_path"], target_key,
    )
    targets_np = np.array(targets)
    data_mean, data_std = float(targets_np.mean()), float(targets_np.std())
    logger.info("Target stats — mean: %.4f  std: %.4f  N=%d", data_mean, data_std, len(targets))

    split = _get_or_create_split(material_ids, train_cfg, data_cfg)
    train_idx, val_idx, test_idx = ids_to_indices(material_ids, split)
    logger.info("Split sizes: train=%d  val=%d  test=%d", len(train_idx), len(val_idx), len(test_idx))

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

    lit_module = RegressionModule(
        model, engine,
        data_mean=data_mean, data_std=data_std,
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    log_dir = train_cfg.get("log_dir", "outputs/logs")
    run_name = f"{arch}_{target_key}"
    csv_logger = CSVLogger(log_dir, name=run_name)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=train_cfg.get("patience", 20), mode="min"),
        ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1,
            filename="best-{epoch:03d}-{val_loss:.4f}",
        ),
    ]

    trainer = L.Trainer(
        max_epochs=train_cfg.get("max_epochs", 100),
        accelerator=train_cfg.get("accelerator", "auto"),
        logger=csv_logger,
        callbacks=callbacks,
        default_root_dir=log_dir,
    )
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_module, dataloaders=test_loader)

    y_pred, y_true = lit_module.get_test_predictions()
    y_pred_np = y_pred.numpy()
    y_true_np = y_true.numpy()

    metrics = compute_metrics(y_true_np, y_pred_np)
    logger.info("Test metrics for %s/%s: %s", arch, target_key, metrics)

    result_dir = Path(log_dir) / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    np.savez(result_dir / "predictions.npz", y_true=y_true_np, y_pred=y_pred_np)

    parity_plot(
        y_true_np, y_pred_np,
        title=f"{arch} — {target_key}",
        metrics=metrics,
        save_path=result_dir / "parity_plot.png",
    )
    error_histogram(
        y_true_np, y_pred_np,
        title=f"{arch} — {target_key}",
        save_path=result_dir / "error_histogram.png",
    )

    with open(result_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return BenchmarkResult(
        property_name=target_key,
        model_name=arch,
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        rmse=metrics["RMSE"],
        mae=metrics["MAE"],
        r2=metrics["R2"],
    )


def _run_descriptor_model(cfg: dict) -> BenchmarkResult | None:
    """Train and evaluate a descriptor-based model (XGBoost or MLP)."""
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    arch = model_cfg.get("arch", "xgboost")
    target_key = data_cfg["target_key"]
    logger.info("Architecture: %s | Target: %s", arch, target_key)

    structures, targets, material_ids = load_structures_and_targets(
        data_cfg["json_path"], target_key,
    )
    targets_np = np.array(targets)
    logger.info("Target stats — mean: %.4f  std: %.4f  N=%d",
                targets_np.mean(), targets_np.std(), len(targets))

    split = _get_or_create_split(material_ids, train_cfg, data_cfg)
    train_idx, val_idx, test_idx = ids_to_indices(material_ids, split)
    logger.info("Split sizes: train=%d  val=%d  test=%d", len(train_idx), len(val_idx), len(test_idx))

    feature_path = data_cfg.get("feature_file")
    if feature_path and Path(feature_path).exists():
        import pandas as pd
        logger.info("Loading pre-computed features from %s", feature_path)
        feat_df = pd.read_parquet(feature_path)
        mid_to_row = {mid: i for i, mid in enumerate(feat_df["material_id"])}
        feat_cols = [c for c in feat_df.columns if c != "material_id"]
        X_all = feat_df[feat_cols].values

        reindex = [mid_to_row[mid] for mid in material_ids if mid in mid_to_row]
        X_all = X_all[reindex]
    else:
        logger.info("Generating matminer features...")
        import pandas as pd
        from matprop_nn.features.matminer_features import MatminerFeaturizer

        feature_groups = model_cfg.get("feature_groups", ["composition", "density"])
        featurizer = MatminerFeaturizer(feature_groups=feature_groups, n_jobs=train_cfg.get("n_jobs", 1))

        records_df = pd.DataFrame({
            "material_id": material_ids,
            "formula_pretty": [s.composition.reduced_formula for s in structures],
            "structure": structures,
        })
        from pymatgen.core import Composition
        records_df["composition"] = [s.composition for s in structures]

        feat_df = featurizer.featurize_df(records_df)
        X_all = feat_df.values

        out_feat = data_cfg.get("feature_output", f"data/processed/features_{target_key}.parquet")
        featurizer.save_features(X_all, material_ids, out_feat)

    from matprop_nn.features.matminer_features import MatminerFeaturizer
    temp_featurizer = MatminerFeaturizer()

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    X_train = pipeline.fit_transform(X_all[train_idx])
    X_val = pipeline.transform(X_all[val_idx])
    X_test = pipeline.transform(X_all[test_idx])

    y_train = targets_np[train_idx]
    y_val = targets_np[val_idx]
    y_test = targets_np[test_idx]

    log_dir = train_cfg.get("log_dir", "outputs/logs")
    run_name = f"{arch}_{target_key}"
    result_dir = Path(log_dir) / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    if arch == "xgboost":
        from matprop_nn.models._xgboost import XGBoostRegressor
        xgb_params = {k: v for k, v in model_cfg.items() if k not in ("arch", "feature_groups")}
        regressor = XGBoostRegressor(**xgb_params)
        regressor.fit(X_train, y_train, X_val, y_val,
                      early_stopping_rounds=train_cfg.get("patience", 30))
        y_pred = regressor.predict(X_test)
        regressor.save(result_dir / "model.json")

    elif arch == "mlp":
        from matprop_nn.models._mlp import MLPRegressor, MLPLitModule
        from matprop_nn.features.descriptor_dataset import DescriptorDataset
        from torch.utils.data import DataLoader

        hidden_dims = model_cfg.get("hidden_dims", [256, 128, 64])
        dropout = model_cfg.get("dropout", 0.1)
        mlp = MLPRegressor(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout)
        lit_mlp = MLPLitModule(
            mlp,
            lr=train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )

        train_ds = DescriptorDataset(X_train, y_train)
        val_ds = DescriptorDataset(X_val, y_val)
        test_ds = DescriptorDataset(X_test, y_test)

        batch_size = train_cfg.get("batch_size", 256)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)
        test_dl = DataLoader(test_ds, batch_size=batch_size)

        csv_logger = CSVLogger(log_dir, name=run_name)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=train_cfg.get("patience", 20), mode="min"),
            ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=1,
                filename="best-{epoch:03d}-{val_loss:.4f}",
            ),
        ]

        trainer = L.Trainer(
            max_epochs=train_cfg.get("max_epochs", 200),
            accelerator=train_cfg.get("accelerator", "auto"),
            logger=csv_logger,
            callbacks=callbacks,
            default_root_dir=log_dir,
        )
        trainer.fit(lit_mlp, train_dataloaders=train_dl, val_dataloaders=val_dl)
        trainer.test(lit_mlp, dataloaders=test_dl)

        preds_list = trainer.predict(lit_mlp, dataloaders=test_dl)
        import torch
        y_pred = torch.cat(preds_list).numpy()
    else:
        raise ValueError(f"Unknown descriptor model: {arch}")

    y_pred = np.asarray(y_pred).ravel()
    metrics = compute_metrics(y_test, y_pred)
    logger.info("Test metrics for %s/%s: %s", arch, target_key, metrics)

    np.savez(result_dir / "predictions.npz", y_true=y_test, y_pred=y_pred)

    parity_plot(
        y_test, y_pred,
        title=f"{arch} — {target_key}",
        metrics=metrics,
        save_path=result_dir / "parity_plot.png",
    )
    error_histogram(
        y_test, y_pred,
        title=f"{arch} — {target_key}",
        save_path=result_dir / "error_histogram.png",
    )

    with open(result_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return BenchmarkResult(
        property_name=target_key,
        model_name=arch,
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        rmse=metrics["RMSE"],
        mae=metrics["MAE"],
        r2=metrics["R2"],
    )


def run(cfg: dict) -> BenchmarkResult | None:
    """Dispatch to graph or descriptor training based on ``model.arch``."""
    arch = cfg["model"].get("arch", "tensornet")
    if is_descriptor_model(arch):
        return _run_descriptor_model(cfg)
    return _run_graph_model(cfg)


def main():
    parser = argparse.ArgumentParser(description="Train a regression model on material data.")
    parser.add_argument("config", type=Path, help="Path to YAML config file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
