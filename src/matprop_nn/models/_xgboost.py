"""XGBoost descriptor-based regressor.

This is not a graph model — it operates on a pre-computed feature matrix
produced by the matminer pipeline.  It does not subclass ModelEngine
because it has a fundamentally different interface (sklearn-style fit/predict
rather than torch DataLoader + forward pass).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class XGBoostRegressor:
    """Thin wrapper around ``xgboost.XGBRegressor`` with project conventions."""

    def __init__(self, **params):
        import xgboost as xgb

        defaults = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(params)
        self.model = xgb.XGBRegressor(**defaults)
        self._params = defaults

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping_rounds: int = 30,
    ):
        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)
        self.model.fit(X_train, y_train, verbose=50, **fit_kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        meta = path.with_suffix(".json")
        with open(meta, "w") as f:
            json.dump(self._params, f, indent=2, default=str)
        logger.info("Saved XGBoost model to %s", path)

    def load(self, path: str | Path):
        import xgboost as xgb
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(path))
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
