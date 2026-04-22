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
from matprop_nn.utils.target_transform import get_target_transform

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _iter_records(path: str | Path):
    """Yield dicts from either JSON (list) or JSONL (one per line).

    The ``.jsonl`` fetch path (full MP summary, ~150k materials) produces a
    file too large to ``json.load`` into memory as a single list, so we
    stream line-by-line.  JSON lists are still supported for the smaller
    dielectric dump.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            yield rec


def load_structures_and_targets(
    json_path: str | Path,
    target_key: str,
) -> tuple[list[Structure], list[float], list[str]]:
    """Read JSON / JSONL, parse pymatgen structures and scalar targets."""
    structures, targets, material_ids = [], [], []
    skipped = 0
    for rec in _iter_records(json_path):
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
    """Load an existing split file or create a new one.

    Ratios can be set either per-task (``data.test_frac`` / ``data.val_frac``)
    or globally (``training.test_frac`` / ``training.val_frac``).  Default
    is 90:10 train:test with an empty validation set — descriptor models
    carve a small internal validation split for early stopping.
    """
    split_path = data_cfg.get("split_file")
    if split_path and Path(split_path).exists():
        existing = load_splits(split_path)
        known_ids = set(existing.get("train", [])) | set(existing.get("val", [])) | set(existing.get("test", []))
        current_ids = set(material_ids)
        overlap = len(known_ids & current_ids)
        # Regenerate if the stored split is for a clearly different dataset
        # (e.g. the dielectric 7k split being reused for the 150k summary run).
        if overlap < 0.5 * len(current_ids):
            logger.warning(
                "Existing split %s only matches %d/%d current material_ids "
                "(stored n_total=%d) — regenerating.",
                split_path, overlap, len(current_ids), existing.get("n_total", -1),
            )
        else:
            logger.info("Loading existing split from %s", split_path)
            return existing

    test_frac = data_cfg.get("test_frac", train_cfg.get("test_frac", 0.10))
    val_frac = data_cfg.get("val_frac", train_cfg.get("val_frac", 0.0))

    split = create_splits(
        material_ids,
        test_frac=test_frac,
        val_frac=val_frac,
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
    logger.info(
        "Target stats — mean: %.4f  std: %.4f  min: %.4f  max: %.4f  N=%d",
        targets_np.mean(), targets_np.std(),
        targets_np.min(), targets_np.max(), len(targets),
    )

    transform_name = (
        data_cfg.get("target_transform")
        or cfg.get("task", {}).get("target_transform")
        or "identity"
    )
    target_transform = get_target_transform(transform_name)
    if target_transform.name != "identity":
        logger.info("Target transform: %s", target_transform.name)
    targets_transformed = target_transform.forward(targets_np)
    data_mean = float(targets_transformed.mean())
    data_std = float(targets_transformed.std())

    split = _get_or_create_split(material_ids, train_cfg, data_cfg)
    train_idx, val_idx, test_idx = ids_to_indices(material_ids, split)
    logger.info("Split sizes: train=%d  val=%d  test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Graph models require a non-empty validation fold for the early-stopping
    # callback (monitor=val_loss).  Descriptor models carve their own val set
    # internally, so the on-disk split commonly has val_frac=0.  In that case
    # deterministically peel off ~10 % of the training indices as val here —
    # no change to the test set, no feature-file regeneration required.
    if len(val_idx) == 0 and len(train_idx) > 0:
        import random as _random
        rng = _random.Random(train_cfg.get("seed", 42))
        shuffled = list(train_idx)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(0.1 * len(shuffled))))
        val_idx = shuffled[:n_val]
        train_idx = shuffled[n_val:]
        logger.info(
            "Graph model: carved val fold from train → train=%d  val=%d  test=%d",
            len(train_idx), len(val_idx), len(test_idx),
        )

    from matgl.ext._pymatgen_pyg import get_element_list
    element_types = get_element_list(structures)

    # Pass transformed targets into the graph dataset: graph models should
    # also optimise in transformed space for comparable benchmarks.
    targets_for_graph = targets_transformed.tolist()
    train_ds, val_ds, test_ds = engine.prepare_datasets(
        structures, targets_for_graph, train_idx, val_idx, test_idx,
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
    y_pred_t = y_pred.numpy()
    y_true_t = y_true.numpy()
    y_pred_orig = target_transform.inverse(y_pred_t)
    y_true_orig = target_transform.inverse(y_true_t)

    # Run a prediction pass over the training set so the parity plot can
    # overlay train vs. test and display both R² values.
    try:
        lit_module.reset_test_predictions()
        trainer.test(lit_module, dataloaders=train_loader, verbose=False)
        y_train_pred_t_tensor, y_train_true_t_tensor = lit_module.get_test_predictions()
        y_train_pred_t = y_train_pred_t_tensor.numpy()
        y_train_true_t = y_train_true_t_tensor.numpy()
        y_train_pred_orig = target_transform.inverse(y_train_pred_t)
        y_train_true_orig = target_transform.inverse(y_train_true_t)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not compute train predictions for parity plot: %s", e)
        y_train_pred_t = y_train_true_t = None
        y_train_pred_orig = y_train_true_orig = None

    metrics_training = compute_metrics(y_true_t, y_pred_t)
    metrics_original = compute_metrics(y_true_orig, y_pred_orig)
    transform_active = target_transform.name != "identity"
    metrics = {
        **metrics_training,
        "training_space": target_transform.name if transform_active else "identity",
        "original_units": metrics_original,
    }

    logger.info(
        "Test metrics (%s space) for %s/%s: RMSE=%.4f  MAE=%.4f  R²=%.4f",
        metrics["training_space"], arch, target_key,
        metrics_training["RMSE"], metrics_training["MAE"], metrics_training["R2"],
    )
    if transform_active:
        logger.info(
            "Test metrics (original units)      : RMSE=%.4f  MAE=%.4f  R²=%.4f",
            metrics_original["RMSE"], metrics_original["MAE"], metrics_original["R2"],
        )

    result_dir = Path(log_dir) / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    npz_payload: dict[str, np.ndarray] = {
        "y_true": y_true_orig, "y_pred": y_pred_orig,
        "y_true_transformed": y_true_t, "y_pred_transformed": y_pred_t,
    }
    if y_train_pred_t is not None:
        npz_payload.update({
            "y_train_true": y_train_true_orig,
            "y_train_pred": y_train_pred_orig,
            "y_train_true_transformed": y_train_true_t,
            "y_train_pred_transformed": y_train_pred_t,
        })
    np.savez(result_dir / "predictions.npz", **npz_payload)

    parity_plot(
        y_true_t, y_pred_t,
        y_train_true=y_train_true_t, y_train_pred=y_train_pred_t,
        title=f"{arch} — {target_key}",
        xlabel="Actual",
        ylabel="Predicted",
        save_path=result_dir / "parity_plot.png",
    )
    error_histogram(
        y_true_t, y_pred_t,
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
    logger.info(
        "Target stats — mean: %.4f  std: %.4f  min: %.4f  max: %.4f  N=%d",
        targets_np.mean(), targets_np.std(),
        targets_np.min(), targets_np.max(), len(targets),
    )

    # Reversible target transform (e.g. log1p for heavy-tailed targets).
    # Training is done in transformed space; metrics are reported in
    # original units via transform.inverse(y_pred).
    transform_name = (
        data_cfg.get("target_transform")
        or cfg.get("task", {}).get("target_transform")
        or "identity"
    )
    target_transform = get_target_transform(transform_name)
    if target_transform.name != "identity":
        logger.info("Target transform: %s", target_transform.name)

    split = _get_or_create_split(material_ids, train_cfg, data_cfg)
    train_idx, val_idx, test_idx = ids_to_indices(material_ids, split)
    logger.info("Split sizes: train=%d  val=%d  test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Always re-assemble features from the per-featurizer cache: the model's
    # ``feature_groups`` setting is authoritative, and the per-featurizer
    # parquet cache makes this effectively free (<1s for 7k rows).  A stale
    # aggregate ``feature_file`` could otherwise leak a mismatched column set
    # (e.g. full 152-feature cache used for a 135-feature XGBoost run).
    import pandas as pd
    from matprop_nn.features.matminer_features import MatminerFeaturizer

    feature_groups = model_cfg.get("feature_groups", ["composition", "density"])
    featurizer_n_jobs = train_cfg.get(
        "featurizer_n_jobs", train_cfg.get("n_jobs", 4)
    )
    feat_cache_dir = data_cfg.get(
        "feature_cache_dir", "data/processed/_feat_cache",
    )
    featurizer = MatminerFeaturizer(
        feature_groups=feature_groups,
        n_jobs=featurizer_n_jobs,
        cache_dir=feat_cache_dir,
    )

    records_df = pd.DataFrame({
        "material_id": material_ids,
        "formula_pretty": [s.composition.reduced_formula for s in structures],
        "structure": structures,
    })
    records_df["composition"] = [s.composition for s in structures]

    feat_df = featurizer.featurize_df(records_df, material_ids=material_ids)
    X_all = feat_df.values
    feature_names: list[str] = list(featurizer.feature_names)

    # Save an aggregate snapshot keyed by feature-group signature so it can
    # be reused by notebooks / external tooling without overwriting other
    # configurations.
    sig = "_".join(sorted({g for g in feature_groups}))
    out_feat = data_cfg.get("feature_output") or (
        f"data/processed/features_{sig}.parquet"
    )
    featurizer.save_features(X_all, material_ids, out_feat)

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # If the split has no validation set (e.g. a 90:10 train:test split),
    # carve an internal val set from train for early stopping.  We keep
    # this deterministic with a fixed seed so runs are reproducible.
    effective_train_idx = list(train_idx)
    effective_val_idx = list(val_idx)
    internal_val_carved = False
    if not effective_val_idx:
        internal_val_frac = train_cfg.get("internal_val_frac", 0.1)
        rng = np.random.default_rng(train_cfg.get("seed", 42))
        perm = rng.permutation(len(effective_train_idx))
        n_val = max(1, int(len(effective_train_idx) * internal_val_frac))
        val_pick = perm[:n_val]
        train_pick = perm[n_val:]
        effective_val_idx = [effective_train_idx[i] for i in val_pick]
        effective_train_idx = [effective_train_idx[i] for i in train_pick]
        internal_val_carved = True
        logger.info(
            "No validation split in file — carved %d (%.0f%%) of train for "
            "early stopping.", len(effective_val_idx), 100 * internal_val_frac,
        )

    X_train = pipeline.fit_transform(X_all[effective_train_idx])
    X_val = pipeline.transform(X_all[effective_val_idx])
    X_test = pipeline.transform(X_all[test_idx])

    y_train_orig = targets_np[effective_train_idx]
    y_val_orig = targets_np[effective_val_idx]
    y_test_orig = targets_np[test_idx]

    y_train = target_transform.forward(y_train_orig)
    y_val = target_transform.forward(y_val_orig)
    y_test = target_transform.forward(y_test_orig)

    log_dir = train_cfg.get("log_dir", "outputs/logs")
    run_name = f"{arch}_{target_key}"
    result_dir = Path(log_dir) / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    if arch == "xgboost":
        from matprop_nn.models._xgboost import XGBoostRegressor
        from matprop_nn.evaluation.feature_importance import (
            compute_xgb_feature_importance,
            save_feature_importance,
        )
        xgb_params = {k: v for k, v in model_cfg.items() if k not in ("arch", "feature_groups")}
        regressor = XGBoostRegressor(**xgb_params)
        regressor.fit(X_train, y_train, X_val, y_val,
                      early_stopping_rounds=train_cfg.get("patience", 30))
        y_pred = regressor.predict(X_test)
        y_train_pred = regressor.predict(X_train)
        regressor.save(result_dir / "model.json")

        # Feature-importance analysis: gain / weight / cover tables + plot.
        importance_df = compute_xgb_feature_importance(
            regressor.model, feature_names,
        )
        top_k = train_cfg.get("feature_importance_top_k", 30)
        save_feature_importance(
            importance_df,
            out_dir=result_dir,
            prefix="feature_importance",
            top_k=top_k,
            title=f"XGBoost feature importance (gain) — {target_key}",
        )
        top5 = importance_df.head(5)
        logger.info(
            "Top-5 features by gain: %s",
            ", ".join(
                f"{r.feature} ({r.gain_normalized:.1%})"
                for r in top5.itertuples()
            ),
        )

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

        import torch
        preds_test = trainer.predict(lit_mlp, dataloaders=test_dl)
        y_pred = torch.cat(preds_test).numpy()
        train_eval_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        preds_train = trainer.predict(lit_mlp, dataloaders=train_eval_dl)
        y_train_pred = torch.cat(preds_train).numpy()
    else:
        raise ValueError(f"Unknown descriptor model: {arch}")

    y_pred_t = np.asarray(y_pred).ravel()
    y_pred = target_transform.inverse(y_pred_t)
    y_train_pred_t = np.asarray(y_train_pred).ravel()
    y_train_pred_orig = target_transform.inverse(y_train_pred_t)

    # Report two metric sets:
    #   * "original" — metrics in the native target units (for human reading).
    #   * "training" — metrics in the space the model actually optimised
    #     (= original when no transform is applied, = log1p for dielectric).
    # For heavy-tailed targets (e.g. dielectric) the original-unit R² is
    # dominated by a handful of DFT outliers, so community practice (Matbench,
    # Petousis 2017, Qu 2020) is to report metrics in log-space.  We use the
    # "training-space" metrics as the primary number in the benchmark table.
    metrics_original = compute_metrics(y_test_orig, y_pred)
    metrics_training = compute_metrics(y_test, y_pred_t)
    transform_active = target_transform.name != "identity"

    metrics = {
        **metrics_training,
        "training_space": target_transform.name if transform_active else "identity",
        "original_units": metrics_original,
    }

    logger.info(
        "Test metrics (%s space) for %s/%s: RMSE=%.4f  MAE=%.4f  R²=%.4f",
        metrics["training_space"], arch, target_key,
        metrics_training["RMSE"], metrics_training["MAE"], metrics_training["R2"],
    )
    if transform_active:
        logger.info(
            "Test metrics (original units)      : RMSE=%.4f  MAE=%.4f  R²=%.4f",
            metrics_original["RMSE"], metrics_original["MAE"], metrics_original["R2"],
        )

    np.savez(
        result_dir / "predictions.npz",
        y_true=y_test_orig, y_pred=y_pred,
        y_true_transformed=y_test, y_pred_transformed=y_pred_t,
        y_train_true=y_train_orig, y_train_pred=y_train_pred_orig,
        y_train_true_transformed=y_train, y_train_pred_transformed=y_train_pred_t,
    )

    parity_plot(
        y_test, y_pred_t,
        y_train_true=y_train, y_train_pred=y_train_pred_t,
        title=f"{arch} — {target_key}",
        xlabel="Actual",
        ylabel="Predicted",
        save_path=result_dir / "parity_plot.png",
    )
    error_histogram(
        y_test, y_pred_t,
        title=f"{arch} — {target_key}",
        save_path=result_dir / "error_histogram.png",
    )

    with open(result_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return BenchmarkResult(
        property_name=target_key,
        model_name=arch,
        n_train=len(effective_train_idx),
        n_val=len(effective_val_idx),
        n_test=len(test_idx),
        rmse=metrics_training["RMSE"],
        mae=metrics_training["MAE"],
        r2=metrics_training["R2"],
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
