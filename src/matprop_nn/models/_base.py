"""ModelEngine base class — abstracts dataset/model/training per architecture."""

from __future__ import annotations

import abc
from typing import Any

import torch
from torch.utils.data import DataLoader

from pymatgen.core import Structure


class ModelEngine(abc.ABC):
    """Each architecture implements this to plug into the unified training loop.

    Subclasses provide four things:
      1. ``prepare_datasets`` — convert structures+targets into train/val/test datasets
      2. ``build_dataloaders``  — wrap datasets in DataLoaders with the right collation
      3. ``build_model``        — instantiate the neural network
      4. ``step``               — forward pass + loss (called by Lightning module)
    """

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def prepare_datasets(
        self,
        structures: list[Structure],
        targets: list[float],
        train_idx: list[int],
        val_idx: list[int],
        test_idx: list[int],
        **kwargs,
    ) -> tuple[Any, Any, Any]:
        """Return ``(train_dataset, val_dataset, test_dataset)``."""

    @abc.abstractmethod
    def build_dataloaders(
        self,
        train_ds,
        val_ds,
        test_ds,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Return ``(train_loader, val_loader, test_loader)``."""

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def build_model(self, element_types: tuple[str, ...], **kwargs) -> torch.nn.Module:
        """Instantiate the architecture."""

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def step(
        self,
        model: torch.nn.Module,
        batch: Any,
        data_mean: float,
        data_std: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Run one forward pass.

        Returns ``(predictions, labels, batch_size)`` where predictions
        are **un-normalised** (original target scale).
        """
