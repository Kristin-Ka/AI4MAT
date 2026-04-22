"""Tabular dataset for descriptor-based models (XGBoost, MLP)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class DescriptorDataset(Dataset):
    """Simple feature-matrix + target dataset for PyTorch MLP training."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
