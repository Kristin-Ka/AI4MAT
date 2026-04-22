"""Model registry — one entry point for any architecture."""

from __future__ import annotations

from matprop_nn.models._base import ModelEngine
from matprop_nn.models._tensornet import TensorNetEngine
from matprop_nn.models._chgnet import CHGNetEngine
from matprop_nn.models._alignn import ALIGNNEngine
from matprop_nn.models._m3gnet import M3GNetEngine
from matprop_nn.models._xgboost import XGBoostRegressor
from matprop_nn.models._mlp import MLPRegressor, MLPLitModule

ENGINES: dict[str, type[ModelEngine]] = {
    "tensornet": TensorNetEngine,
    "chgnet": CHGNetEngine,
    "alignn": ALIGNNEngine,
    "m3gnet": M3GNetEngine,
}

DESCRIPTOR_MODELS = {"xgboost", "mlp"}
GRAPH_MODELS = set(ENGINES.keys())
ALL_MODELS = GRAPH_MODELS | DESCRIPTOR_MODELS
AVAILABLE_ARCHS = sorted(ALL_MODELS)


def get_engine(arch: str) -> ModelEngine:
    """Instantiate a :class:`ModelEngine` by architecture name."""
    if arch not in ENGINES:
        raise ValueError(
            f"Unknown graph architecture '{arch}'. Available: {sorted(ENGINES)}"
        )
    return ENGINES[arch]()


def is_descriptor_model(arch: str) -> bool:
    return arch in DESCRIPTOR_MODELS


def build_regressor(element_types, **kwargs):
    """Shortcut that builds a TensorNet regressor (backward compat)."""
    return TensorNetEngine().build_model(element_types, **kwargs)


__all__ = [
    "ModelEngine",
    "TensorNetEngine",
    "CHGNetEngine",
    "ALIGNNEngine",
    "M3GNetEngine",
    "XGBoostRegressor",
    "MLPRegressor",
    "MLPLitModule",
    "ENGINES",
    "DESCRIPTOR_MODELS",
    "GRAPH_MODELS",
    "ALL_MODELS",
    "AVAILABLE_ARCHS",
    "get_engine",
    "is_descriptor_model",
    "build_regressor",
]
