"""Model registry — one entry point for any architecture."""

from __future__ import annotations

from matprop_nn.models._base import ModelEngine
from matprop_nn.models._tensornet import TensorNetEngine
from matprop_nn.models._chgnet import CHGNetEngine
from matprop_nn.models._alignn import ALIGNNEngine

ENGINES: dict[str, type[ModelEngine]] = {
    "tensornet": TensorNetEngine,
    "chgnet": CHGNetEngine,
    "alignn": ALIGNNEngine,
}

AVAILABLE_ARCHS = list(ENGINES.keys())


def get_engine(arch: str) -> ModelEngine:
    """Instantiate a :class:`ModelEngine` by architecture name.

    >>> engine = get_engine("tensornet")
    """
    if arch not in ENGINES:
        raise ValueError(
            f"Unknown architecture '{arch}'. Available: {AVAILABLE_ARCHS}"
        )
    return ENGINES[arch]()


# Keep backward-compatible shortcut
def build_regressor(element_types, **kwargs):
    """Shortcut that builds a TensorNet regressor (backward compat)."""
    return TensorNetEngine().build_model(element_types, **kwargs)


__all__ = [
    "ModelEngine",
    "TensorNetEngine",
    "CHGNetEngine",
    "ALIGNNEngine",
    "ENGINES",
    "AVAILABLE_ARCHS",
    "get_engine",
    "build_regressor",
]
