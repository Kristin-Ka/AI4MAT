"""Reversible target transforms for heavy-tailed regression targets.

Many materials properties (dielectric constant, band gap, energy above hull)
have strongly skewed distributions — a small number of extreme values dominate
MSE loss and cause models to generalise worse than the mean.  The standard
remedy in materials ML literature is to train on ``log1p`` or ``log10`` of the
target and report metrics back in the original units.

Usage::

    transform = get_target_transform("log1p")
    y_train_t = transform.forward(y_train)       # train on transformed
    # ... fit model ...
    y_pred = transform.inverse(y_pred_t)         # back to original units
    metrics = compute_metrics(y_true, y_pred)    # metrics in original units

Available transforms:
  * ``identity``       — no transform
  * ``log1p``          — ``y → log(1 + y)``; requires ``y ≥ 0``
  * ``log10_positive`` — ``y → log10(max(y, eps))``; for strictly positive y
  * ``signed_log1p``   — ``y → sign(y) * log(1 + |y|)``; handles negative y
"""

from __future__ import annotations

import numpy as np


class TargetTransform:
    """Base class — identity transform."""

    name = "identity"

    def forward(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float)

    def inverse(self, y_t: np.ndarray) -> np.ndarray:
        return np.asarray(y_t, dtype=float)

    def __repr__(self) -> str:
        return f"TargetTransform(name={self.name!r})"


class Log1pTransform(TargetTransform):
    """``y → log(1 + y)``.  Requires y ≥ 0 (enforced by clipping)."""

    name = "log1p"

    def forward(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            # Clip rather than fail — a few slightly-negative targets can arise
            # from numerical noise (e.g. e_hull ≈ -1e-5).
            y = np.clip(y, 0.0, None)
        return np.log1p(y)

    def inverse(self, y_t: np.ndarray) -> np.ndarray:
        y_t = np.asarray(y_t, dtype=float)
        return np.expm1(y_t)


class Log10PositiveTransform(TargetTransform):
    """``y → log10(max(y, eps))``.  Suitable for strictly-positive targets."""

    name = "log10_positive"

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def forward(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return np.log10(np.clip(y, self.eps, None))

    def inverse(self, y_t: np.ndarray) -> np.ndarray:
        y_t = np.asarray(y_t, dtype=float)
        return np.power(10.0, y_t)


class SignedLog1pTransform(TargetTransform):
    """``y → sign(y) * log(1 + |y|)``.  Preserves sign for mixed-sign targets."""

    name = "signed_log1p"

    def forward(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return np.sign(y) * np.log1p(np.abs(y))

    def inverse(self, y_t: np.ndarray) -> np.ndarray:
        y_t = np.asarray(y_t, dtype=float)
        return np.sign(y_t) * np.expm1(np.abs(y_t))


_REGISTRY: dict[str, type[TargetTransform]] = {
    "identity": TargetTransform,
    "none": TargetTransform,
    "log1p": Log1pTransform,
    "log10": Log10PositiveTransform,
    "log10_positive": Log10PositiveTransform,
    "signed_log1p": SignedLog1pTransform,
}


def get_target_transform(name: str | None) -> TargetTransform:
    """Factory — returns a :class:`TargetTransform` by short name."""
    if name is None:
        return TargetTransform()
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown target_transform '{name}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]()
