"""ALIGNN engine — Atomistic Line Graph Neural Network.

Supports both training from scratch and fine-tuning a pretrained ALIGNN
backbone (e.g. ``pretrained="jv_formation_energy_peratom_alignn"``).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

import dgl
from pymatgen.core import Structure
from jarvis.core.atoms import Atoms as JarvisAtoms
from alignn.graphs import Graph as ALIGNNGraph
from alignn.models.alignn import ALIGNN, ALIGNNConfig

from ._base import ModelEngine

logger = logging.getLogger(__name__)


# ── Helpers -------------------------------------------------------------------

def _pymatgen_to_jarvis(struct: Structure) -> JarvisAtoms:
    return JarvisAtoms(
        lattice_mat=struct.lattice.matrix.tolist(),
        coords=struct.frac_coords.tolist(),
        elements=[str(s) for s in struct.species],
        cartesian=False,
    )


class _ALIGNNRegressor(nn.Module):
    """Pretrained ALIGNN backbone + fresh scalar regression head."""

    def __init__(self, backbone: ALIGNN, hidden: int):
        super().__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        return self.head(feat).squeeze(-1)


# ── Dataset -------------------------------------------------------------------

class _ALIGNNDataset(Dataset):
    """Pre-computed DGL (g, lg, lattice) triples with scalar targets."""

    def __init__(self, graph_triples, targets):
        self.triples = graph_triples
        self.targets = targets

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx], self.targets[idx]


def _alignn_collate(batch):
    """Collate (g, lg, lat) triples into batched DGL graphs."""
    triples, targets = zip(*batch)
    gs, lgs, lats = zip(*triples)
    batched_g = dgl.batch(list(gs))
    batched_lg = dgl.batch(list(lgs))
    batched_lat = torch.stack(list(lats))
    return (batched_g, batched_lg, batched_lat), torch.tensor(
        targets, dtype=torch.float32
    )


# ── Engine --------------------------------------------------------------------

class ALIGNNEngine(ModelEngine):

    def prepare_datasets(
        self,
        structures: list[Structure],
        targets: list[float],
        train_idx: list[int],
        val_idx: list[int],
        test_idx: list[int],
        *,
        cutoff: float = 8.0,
        max_neighbors: int = 12,
        **kwargs,
    ):
        logger.info("Converting %d structures to ALIGNN DGL graphs...", len(structures))
        triples = []
        valid_targets = []
        valid_global_idx = []
        for i, struct in enumerate(structures):
            try:
                atoms = _pymatgen_to_jarvis(struct)
                g, lg = ALIGNNGraph.atom_dgl_multigraph(
                    atoms,
                    cutoff=cutoff,
                    max_neighbors=max_neighbors,
                    compute_line_graph=True,
                )
                lat = torch.tensor(struct.lattice.matrix, dtype=torch.float32)
                triples.append((g, lg, lat))
                valid_targets.append(targets[i])
                valid_global_idx.append(i)
            except Exception:
                logger.warning("Structure %d failed ALIGNN conversion, skipping.", i)

        idx_set = set(valid_global_idx)
        local_map = {g: l for l, g in enumerate(valid_global_idx)}

        def _remap(idx_list):
            return [local_map[i] for i in idx_list if i in idx_set]

        full_ds = _ALIGNNDataset(triples, valid_targets)
        return (
            Subset(full_ds, _remap(train_idx)),
            Subset(full_ds, _remap(val_idx)),
            Subset(full_ds, _remap(test_idx)),
        )

    def build_dataloaders(
        self, train_ds, val_ds, test_ds, *, batch_size=32, num_workers=0,
    ):
        kw = dict(collate_fn=_alignn_collate, num_workers=num_workers)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw),
        )

    def build_model(self, element_types, **kwargs):
        pretrained = kwargs.get("pretrained")

        if pretrained:
            logger.info("Loading pretrained ALIGNN: %s", pretrained)
            from alignn.pretrained import get_figshare_model
            backbone = get_figshare_model(pretrained)
            hidden = backbone.config.hidden_features
            model = _ALIGNNRegressor(backbone, hidden)
            logger.info(
                "ALIGNN pretrained backbone (%d params, hidden=%d) + fresh head",
                sum(p.numel() for p in backbone.parameters()), hidden,
            )
            return model

        config = ALIGNNConfig(
            name="alignn",
            alignn_layers=kwargs.get("alignn_layers", 4),
            gcn_layers=kwargs.get("gcn_layers", 4),
            atom_input_features=92,
            edge_input_features=80,
            embedding_features=kwargs.get("units", 64),
            hidden_features=kwargs.get("hidden_features", 256),
            output_features=1,
        )
        return ALIGNN(config)

    def step(self, model, batch, data_mean, data_std):
        (batched_g, batched_lg, batched_lat), labels = batch
        device = next(model.parameters()).device
        batched_g = batched_g.to(device)
        batched_lg = batched_lg.to(device)
        batched_lat = batched_lat.to(device)
        labels = labels.to(device)

        preds = model((batched_g, batched_lg, batched_lat))
        preds = preds.squeeze(-1) if preds.dim() > 1 else preds
        return preds, labels, preds.numel()
