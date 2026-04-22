"""MLP (fully connected neural network) for descriptor-based regression.

Operates on pre-computed feature vectors from the matminer pipeline.
Uses a PyTorch Lightning training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import lightning as L
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score


class MLPRegressor(nn.Module):
    """Multi-layer perceptron for tabular regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPLitModule(L.LightningModule):
    """Lightning wrapper for MLP descriptor regression."""

    def __init__(
        self,
        model: MLPRegressor,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

        self.loss_fn = nn.MSELoss()
        self.mae = MeanAbsoluteError()
        self.rmse = MeanSquaredError(squared=False)
        self.r2 = R2Score()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        self.log(f"{stage}_loss", loss, batch_size=len(y), prog_bar=True)
        self.log(f"{stage}_MAE", self.mae(pred, y), batch_size=len(y), prog_bar=True)
        self.log(f"{stage}_RMSE", self.rmse(pred, y), batch_size=len(y))
        self.log(f"{stage}_R2", self.r2(pred, y), batch_size=len(y))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=self.lr * 0.01,
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}
