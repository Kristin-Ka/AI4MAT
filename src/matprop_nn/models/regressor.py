"""Model factory for scalar regression on crystal graphs.

MatGL v2.x (PyG backend) ships M3GNet as DGL-only.  TensorNet is the
PyG-native architecture with comparable expressiveness, so we default to it.
"""

from __future__ import annotations

from matgl.models import TensorNet


def build_regressor(
    element_types: tuple[str, ...],
    *,
    cutoff: float = 5.0,
    units: int = 64,
    nblocks: int = 3,
    num_rbf: int = 32,
    readout_type: str = "weighted_atom",
    is_intensive: bool = True,
    ntargets: int = 1,
    activation_type: str = "swish",
    **kwargs,
) -> TensorNet:
    """Instantiate a TensorNet configured for scalar regression.

    All extra keyword arguments are forwarded to :class:`matgl.models.TensorNet`.
    """
    return TensorNet(
        element_types=element_types,
        units=units,
        nblocks=nblocks,
        num_rbf=num_rbf,
        cutoff=cutoff,
        readout_type=readout_type,
        is_intensive=is_intensive,
        ntargets=ntargets,
        task_type="regression",
        activation_type=activation_type,
        **kwargs,
    )
