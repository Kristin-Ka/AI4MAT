"""CHGNet engine — Crystal Hamiltonian Graph Neural Network.

Supports both training from scratch and fine-tuning the official pretrained
CHGNet backbone (412K params, trained on 1.5M MP structures) via
``pretrained=True``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from pymatgen.core import Structure
from chgnet.graph import CrystalGraphConverter
from chgnet.model import CHGNet

from ._base import ModelEngine

logger = logging.getLogger(__name__)


# ── Dataset ------------------------------------------------------------------

class _CHGNetDataset(Dataset):
    """Stores pre-computed ``CrystalGraph`` objects alongside scalar targets."""

    def __init__(self, crystal_graphs, targets):
        self.graphs = crystal_graphs
        self.targets = targets

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]


def _chgnet_collate(batch):
    """Collate a list of (CrystalGraph, target) into batched tensors."""
    graphs, targets = zip(*batch)
    return list(graphs), torch.tensor(targets, dtype=torch.float32)


# ── Wrapper model ------------------------------------------------------------

class CHGNetRegressor(nn.Module):
    """Wraps ``chgnet.model.CHGNet`` for scalar regression.

    Extracts per-crystal features and passes them through a fresh MLP head
    so the target can be any scalar property, not just formation energy.
    """

    def __init__(self, backbone: CHGNet, hidden_dim: int = 64):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graphs: list):
        out = self.backbone(graphs, task="e", return_crystal_feas=True)
        crystal_fea = out["crystal_fea"]
        return self.head(crystal_fea).squeeze(-1)


# ── Engine -------------------------------------------------------------------

class CHGNetEngine(ModelEngine):

    def prepare_datasets(
        self,
        structures: list[Structure],
        targets: list[float],
        train_idx: list[int],
        val_idx: list[int],
        test_idx: list[int],
        *,
        atom_graph_cutoff: float = 6.0,
        bond_graph_cutoff: float = 3.0,
        **kwargs,
    ):
        converter = CrystalGraphConverter(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            on_isolated_atoms="warn",
        )
        logger.info("Converting %d structures to CHGNet CrystalGraphs...", len(structures))
        graphs = []
        valid_targets = []
        valid_global_idx = []
        for i, struct in enumerate(structures):
            try:
                g = converter(struct)
                graphs.append(g)
                valid_targets.append(targets[i])
                valid_global_idx.append(i)
            except Exception:
                logger.warning("Structure %d failed CHGNet conversion, skipping.", i)

        idx_set = set(valid_global_idx)
        local_map = {g: l for l, g in enumerate(valid_global_idx)}

        def _remap(idx_list):
            return [local_map[i] for i in idx_list if i in idx_set]

        full_ds = _CHGNetDataset(graphs, valid_targets)
        return (
            Subset(full_ds, _remap(train_idx)),
            Subset(full_ds, _remap(val_idx)),
            Subset(full_ds, _remap(test_idx)),
        )

    def build_dataloaders(
        self, train_ds, val_ds, test_ds, *, batch_size=32, num_workers=0,
    ):
        kw = dict(collate_fn=_chgnet_collate, num_workers=num_workers)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw),
        )

    def build_model(self, element_types, **kwargs):
        pretrained = kwargs.get("pretrained", False)
        hidden = kwargs.get("units", 64)

        if pretrained:
            logger.info("Loading pretrained CHGNet backbone...")
            backbone = CHGNet.load(use_device="cpu", verbose=False)
            hidden = backbone.atom_fea_dim
            logger.info(
                "CHGNet pretrained backbone loaded (%d params, atom_fea_dim=%d)",
                sum(p.numel() for p in backbone.parameters()), hidden,
            )
        else:
            backbone = CHGNet(
                atom_fea_dim=hidden,
                bond_fea_dim=hidden,
                angle_fea_dim=hidden,
                num_radial=kwargs.get("num_rbf", 31),
                num_angular=kwargs.get("num_angular", 31),
                n_conv=kwargs.get("nblocks", 3),
                composition_model=None,
            )

        return CHGNetRegressor(backbone, hidden_dim=hidden)

    def step(self, model, batch, data_mean, data_std):
        graphs, labels = batch
        device = next(model.parameters()).device
        graphs = [g.to(device) for g in graphs]
        labels = labels.to(device)
        preds = model(graphs)
        return preds, labels, preds.numel()
