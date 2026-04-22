"""M3GNet engine — multi-body interaction graph neural network via matgl (DGL).

M3GNet in matgl v2 requires the DGL backend. This engine sets the backend
before importing M3GNet-specific modules, converts structures to DGL graphs
with the matgl DGL converter, and wraps the model for scalar property regression.

Also available via DGL backend: MEGNet, SO3Net.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from pymatgen.core import Structure

from ._base import ModelEngine

logger = logging.getLogger(__name__)


_DGL_READY = False


def _ensure_dgl_backend():
    """Switch matgl to DGL backend and reload dependent modules if needed."""
    global _DGL_READY
    if _DGL_READY:
        return
    import importlib
    import matgl
    from matgl import config as matgl_config
    if getattr(matgl_config, "BACKEND", None) != "DGL":
        matgl.set_backend("DGL")
        import matgl.layers
        importlib.reload(matgl.layers)
        import matgl.models
        importlib.reload(matgl.models)
    _DGL_READY = True


class M3GNetEngine(ModelEngine):
    """Uses matgl's ``M3GNet`` (DGL backend) for scalar regression."""

    def prepare_datasets(
        self,
        structures: list[Structure],
        targets: list[float],
        train_idx: list[int],
        val_idx: list[int],
        test_idx: list[int],
        *,
        cutoff: float = 5.0,
        element_types: tuple[str, ...] | None = None,
        cache_dir: str = "MGLDataset_m3gnet",
        target_key: str = "target",
        **kwargs,
    ):
        _ensure_dgl_backend()
        from matgl.ext.pymatgen import Structure2Graph, get_element_list
        from matgl.graph.data import MGLDataset

        if element_types is None:
            element_types = get_element_list(structures)
        self._element_types = element_types

        converter = Structure2Graph(element_types=element_types, cutoff=cutoff)

        dataset = MGLDataset(
            structures=structures,
            labels={target_key: targets},
            converter=converter,
            directory_name=cache_dir,
            save_cache=True,
        )
        return (
            Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx),
        )

    def build_dataloaders(
        self, train_ds, val_ds, test_ds, *, batch_size=32, num_workers=0,
    ):
        _ensure_dgl_backend()
        from matgl.graph.data import MGLDataLoader, collate_fn_graph

        return MGLDataLoader(
            train_data=train_ds,
            val_data=val_ds,
            test_data=test_ds,
            collate_fn=collate_fn_graph,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def build_model(self, element_types, **kwargs):
        _ensure_dgl_backend()
        from matgl.models import M3GNet

        return M3GNet(
            element_types=element_types,
            is_intensive=kwargs.get("is_intensive", True),
            readout_type=kwargs.get("readout_type", "weighted_atom"),
            ntargets=kwargs.get("ntargets", 1),
            cutoff=kwargs.get("cutoff", 5.0),
            units=kwargs.get("units", 64),
            nblocks=kwargs.get("nblocks", 3),
            task_type="regression",
        )

    def step(self, model, batch, data_mean, data_std):
        import dgl

        g, lat, state_attr, labels = batch
        device = lat.device

        g = g.to(device)

        num_edges_per = g.batch_num_edges()
        num_nodes_per = g.batch_num_nodes()

        lattice = torch.repeat_interleave(lat, num_edges_per, dim=0)
        g.edata["pbc_offshift"] = (
            g.edata["pbc_offset"].unsqueeze(dim=-1) * lattice
        ).sum(dim=1)

        g.ndata["pos"] = (
            g.ndata["frac_coords"].unsqueeze(dim=-1)
            * torch.repeat_interleave(lat, num_nodes_per, dim=0)
        ).sum(dim=1)

        preds = model(g, state_attr=state_attr)
        return preds, labels, preds.numel()
