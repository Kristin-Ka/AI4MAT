"""TensorNet engine — matgl's PyG-native graph neural network."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from pymatgen.core import Structure
from matgl.ext._pymatgen_pyg import Structure2Graph, get_element_list
from matgl.graph._data_pyg import MGLDataLoader, MGLDataset, collate_fn_graph
from matgl.models import TensorNet
from torch_geometric.data import Batch, Data

from ._base import ModelEngine

logger = logging.getLogger(__name__)


def _batch_num_edges(g: Batch | Data) -> torch.Tensor:
    slices = g._slice_dict["edge_index"]
    return slices[1:] - slices[:-1]


def _batch_num_nodes(g: Batch | Data) -> torch.Tensor:
    return g.ptr[1:] - g.ptr[:-1]



class TensorNetEngine(ModelEngine):
    """Uses matgl's ``TensorNet`` with ``Structure2Graph`` (PyG backend)."""

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
        cache_dir: str = "MGLDataset",
        target_key: str = "target",
        **kwargs,
    ):
        if element_types is None:
            element_types = get_element_list(structures)
        self._element_types = element_types

        converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
        dataset = MGLDataset(
            structures=structures,
            labels={target_key: targets},
            converter=converter,
            root=cache_dir,
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
        return MGLDataLoader(
            train_data=train_ds,
            val_data=val_ds,
            test_data=test_ds,
            collate_fn=collate_fn_graph,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def build_model(self, element_types, **kwargs):
        return TensorNet(
            element_types=element_types,
            units=kwargs.get("units", 64),
            nblocks=kwargs.get("nblocks", 3),
            num_rbf=kwargs.get("num_rbf", 32),
            cutoff=kwargs.get("cutoff", 5.0),
            readout_type=kwargs.get("readout_type", "weighted_atom"),
            is_intensive=kwargs.get("is_intensive", True),
            ntargets=kwargs.get("ntargets", 1),
            task_type="regression",
            activation_type=kwargs.get("activation_type", "swish"),
        )

    def step(self, model, batch, data_mean, data_std):
        g, lat, state_attr, labels = batch
        device = lat.device

        num_edges = _batch_num_edges(g).to(device)
        num_nodes = _batch_num_nodes(g).to(device)

        lattice = torch.repeat_interleave(lat, num_edges, dim=0)
        g.pbc_offshift = (g.pbc_offset.unsqueeze(dim=-1) * lattice).sum(dim=1)
        g.pos = (
            g.frac_coords.unsqueeze(dim=-1)
            * torch.repeat_interleave(lat, num_nodes, dim=0)
        ).sum(dim=1)

        preds = model(g, state_attr=state_attr)
        return preds, labels, preds.numel()
