"""Hyperparameter sweep for the XGBoost descriptor model.

Runs a lean grid over the most impactful XGBoost knobs for tabular
regression:

    - n_estimators            (max rounds; early stopping cuts it short)
    - max_depth               (tree complexity)
    - learning_rate           (shrinkage)
    - min_child_weight        (regularisation via leaf mass)
    - gamma                   (minimum loss reduction per split)
    - subsample               (row bagging)
    - colsample_bytree        (feature bagging)
    - reg_alpha / reg_lambda  (L1 / L2)

Uses the *same* 90:10 split as the benchmark (via the task config) and
an internal 10%-of-train validation fold carved deterministically for
early stopping.  Reports test-set R² / RMSE in the training space (the
space the model actually optimises), because that is where the label
transformation is defined in `configs/tasks/*.yaml` (e.g. ``log1p`` for
dielectric constants).

Results are written to ``<log_dir>/sweep/results.csv`` and the best
parameter set is printed as a YAML snippet ready to paste into
``configs/models/xgboost.yaml``.

Usage:
    python scripts/sweep_xgboost.py --task configs/tasks/total_dielectric.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matprop_nn  # noqa: F401  (triggers torch preload)

from matprop_nn.tasks.train import load_structures_and_targets
from matprop_nn.evaluation.metrics import compute_metrics
from matprop_nn.features.matminer_features import MatminerFeaturizer
from matprop_nn.models._xgboost import XGBoostRegressor
from matprop_nn.utils.config import load_config
from matprop_nn.utils.splits import create_splits, ids_to_indices, load_splits, save_splits
from matprop_nn.utils.target_transform import get_target_transform

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sweep_xgboost")


# --------------------------- search space --------------------------- #
# Kept deliberately small (108 configs) so the sweep finishes in a few
# minutes on CPU.  Widen it by adding values once a promising region is
# identified.
SEARCH_SPACE: dict[str, list] = {
    "max_depth":        [5, 7, 9],
    "learning_rate":    [0.03, 0.05, 0.08],
    "min_child_weight": [1, 3],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.7, 0.9],
}
# Less-impactful knobs kept at tuned defaults (no combinatorial cost).
DEFAULT_REG = {
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

FIXED_PARAMS = {
    "n_estimators": 3000,        # relies on early stopping
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


def _iter_candidates(space: dict[str, list]) -> list[dict]:
    keys = list(space.keys())
    grid = list(itertools.product(*(space[k] for k in keys)))
    return [dict(zip(keys, combo)) for combo in grid]


def _prepare_features_and_targets(task_cfg: dict, model_cfg: dict, train_cfg: dict):
    data_cfg = task_cfg["data"]
    target_key = task_cfg["task"]["target_key"]
    feature_groups = model_cfg.get("feature_groups", ["elem_prop", "density"])

    structures, targets, material_ids = load_structures_and_targets(
        data_cfg["json_path"], target_key,
    )
    targets_np = np.array(targets)

    transform_name = (
        task_cfg["task"].get("target_transform")
        or data_cfg.get("target_transform")
        or "identity"
    )
    target_transform = get_target_transform(transform_name)
    targets_t = target_transform.forward(targets_np)

    featurizer = MatminerFeaturizer(
        feature_groups=feature_groups,
        n_jobs=train_cfg.get("featurizer_n_jobs", 4),
        cache_dir=Path("data/processed/_feat_cache"),
    )
    import pandas as pd
    records_df = pd.DataFrame({
        "material_id": material_ids,
        "formula_pretty": [s.composition.reduced_formula for s in structures],
        "structure": structures,
    })
    records_df["composition"] = [s.composition for s in structures]
    feat_df = featurizer.featurize_df(records_df, material_ids=material_ids)
    X_all = feat_df.values
    feature_names = list(featurizer.feature_names)
    logger.info("Feature matrix: %s  (%d features)", X_all.shape, len(feature_names))

    # 90:10 split (val carved internally for early stopping).
    test_frac = data_cfg.get("test_frac", 0.10)
    val_frac = data_cfg.get("val_frac", 0.0)
    split_path = Path(data_cfg.get("split_file", f"data/splits/split_{target_key}.json"))
    if split_path.exists():
        split = load_splits(split_path)
    else:
        split = create_splits(
            material_ids, test_frac=test_frac, val_frac=val_frac,
            seed=train_cfg.get("seed", 42),
        )
        save_splits(split, split_path)
    train_idx, val_idx, test_idx = ids_to_indices(material_ids, split)

    # Carve an internal val fold from train (matches _run_descriptor_model).
    if not val_idx:
        rng = np.random.default_rng(train_cfg.get("seed", 42))
        perm = rng.permutation(len(train_idx))
        n_val = max(1, int(len(train_idx) * 0.1))
        val_idx = [train_idx[i] for i in perm[:n_val]]
        train_idx = [train_idx[i] for i in perm[n_val:]]

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train = pipeline.fit_transform(X_all[train_idx])
    X_val = pipeline.transform(X_all[val_idx])
    X_test = pipeline.transform(X_all[test_idx])
    y_train = targets_t[train_idx]
    y_val = targets_t[val_idx]
    y_test_t = targets_t[test_idx]
    y_test_orig = targets_np[test_idx]

    return dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test_t=y_test_t, y_test_orig=y_test_orig,
        target_transform=target_transform,
        target_key=target_key,
        feature_groups=feature_groups,
    )


def run_sweep(task_path: str, model_path: str, out_dir: str, early_stopping_rounds: int = 30):
    task_cfg = load_config(task_path)
    model_cfg = load_config(model_path)["model"]
    train_cfg = load_config(model_path).get("training", {})

    prep = _prepare_features_and_targets(task_cfg, model_cfg, train_cfg)
    X_train, y_train = prep["X_train"], prep["y_train"]
    X_val, y_val = prep["X_val"], prep["y_val"]
    X_test, y_test_t, y_test_orig = prep["X_test"], prep["y_test_t"], prep["y_test_orig"]
    target_transform = prep["target_transform"]

    candidates = _iter_candidates(SEARCH_SPACE)
    logger.info(
        "Sweeping %d candidates  (train=%d  val=%d  test=%d)",
        len(candidates), len(X_train), len(X_val), len(X_test),
    )

    records = []
    for i, cand in enumerate(candidates, 1):
        params = {**FIXED_PARAMS, **DEFAULT_REG, **cand}
        reg = XGBoostRegressor(**params)
        reg.fit(X_train, y_train, X_val, y_val,
                early_stopping_rounds=early_stopping_rounds)

        y_pred_t = reg.predict(X_test)
        y_pred = target_transform.inverse(y_pred_t)

        m_train = compute_metrics(y_test_t, y_pred_t)         # training space
        m_orig = compute_metrics(y_test_orig, y_pred)          # original units
        best_iter = int(getattr(reg.model, "best_iteration", params["n_estimators"]) or 0)

        rec = {
            **cand,
            "best_iter": best_iter,
            "R2_train_space": m_train["R2"],
            "RMSE_train_space": m_train["RMSE"],
            "MAE_train_space": m_train["MAE"],
            "R2_original": m_orig["R2"],
            "RMSE_original": m_orig["RMSE"],
        }
        records.append(rec)
        logger.info(
            "[%3d/%d] R²(log)=%.4f  R²(orig)=%.4f  best_iter=%d  %s",
            i, len(candidates),
            m_train["R2"], m_orig["R2"], best_iter,
            {k: cand[k] for k in ("max_depth", "learning_rate", "subsample", "colsample_bytree")},
        )

    df = pd.DataFrame.from_records(records)
    df.sort_values("R2_train_space", ascending=False, inplace=True)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir_path / "results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved sweep results to %s", csv_path)

    best = df.iloc[0].to_dict()
    logger.info("\n===== BEST CONFIG =====\n%s", json.dumps(best, indent=2, default=str))

    yaml_snippet = (
        "model:\n"
        "  arch: \"xgboost\"\n"
        f"  feature_groups: {prep['feature_groups']}\n"
        f"  n_estimators: {FIXED_PARAMS['n_estimators']}\n"
        f"  max_depth: {int(best['max_depth'])}\n"
        f"  learning_rate: {best['learning_rate']}\n"
        f"  min_child_weight: {int(best['min_child_weight'])}\n"
        f"  subsample: {best['subsample']}\n"
        f"  colsample_bytree: {best['colsample_bytree']}\n"
        f"  gamma: {DEFAULT_REG['gamma']}\n"
        f"  reg_alpha: {DEFAULT_REG['reg_alpha']}\n"
        f"  reg_lambda: {DEFAULT_REG['reg_lambda']}\n"
    )
    (out_dir_path / "best_model.yaml").write_text(yaml_snippet)
    logger.info("Best YAML snippet:\n%s", yaml_snippet)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="configs/tasks/total_dielectric.yaml")
    p.add_argument("--model", default="configs/models/xgboost.yaml")
    p.add_argument("--out", default="results/xgboost/sweep")
    p.add_argument("--early-stopping-rounds", type=int, default=30)
    args = p.parse_args()
    run_sweep(args.task, args.model, args.out, args.early_stopping_rounds)


if __name__ == "__main__":
    main()
