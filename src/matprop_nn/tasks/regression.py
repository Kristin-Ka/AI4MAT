"""Generic Lightning module for scalar regression.

Works with *any* ``ModelEngine`` — the engine's ``step()`` handles
architecture-specific batch unpacking and forward pass.

Supports optional log-transform of targets: when ``log_target=True``,
targets are ``log1p``-transformed before training and predictions are
``expm1``-inverted for metrics, which stabilises training on heavily
skewed distributions (e.g. dielectric constants).
"""

from __future__ import annotations

from typing import Any

import torch
import lightning as L
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from matprop_nn.models._base import ModelEngine


class RegressionModule(L.LightningModule):
    """Architecture-agnostic regression training loop."""

    def __init__(
        self,
        model: torch.nn.Module,
        engine: ModelEngine,
        *,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        log_target: bool = False,
    ):
        super().__init__()
        self.model = model
        self.engine = engine
        self.data_mean = data_mean
        self.data_std = data_std
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_target = log_target

        self.loss_fn = torch.nn.MSELoss()
        self.mae = MeanAbsoluteError()
        self.rmse = MeanSquaredError(squared=False)
        self.r2 = R2Score()

        self._test_preds: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []

    def forward(self, batch):
        preds, labels, _ = self.engine.step(
            self.model, batch, self.data_mean, self.data_std,
        )
        return preds, labels

    def _shared_step(self, batch, stage: str):
        preds, labels, bs = self.engine.step(
            self.model, batch, self.data_mean, self.data_std,
        )
        loss = self.loss_fn(preds, labels)
        self.log(f"{stage}_loss", loss, batch_size=bs, prog_bar=True)

        if self.log_target:
            preds_orig = torch.expm1(preds.detach())
            labels_orig = torch.expm1(labels.detach())
        else:
            preds_orig = preds.detach()
            labels_orig = labels.detach()

        mae = self.mae(preds_orig, labels_orig)
        rmse = self.rmse(preds_orig, labels_orig)
        r2 = self.r2(preds_orig, labels_orig)
        self.log(f"{stage}_MAE", mae, batch_size=bs, prog_bar=True)
        self.log(f"{stage}_RMSE", rmse, batch_size=bs, prog_bar=False)
        self.log(f"{stage}_R2", r2, batch_size=bs, prog_bar=False)

        if stage == "test":
            self._test_preds.append(preds_orig.cpu())
            self._test_labels.append(labels_orig.cpu())

        return loss

    def on_test_epoch_start(self):
        self._test_preds = []
        self._test_labels = []

    def get_test_predictions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (all_preds, all_labels) collected during test."""
        return torch.cat(self._test_preds), torch.cat(self._test_labels)

    def reset_test_predictions(self) -> None:
        """Clear cached test predictions (used to re-use the test hook on
        the training loader to get a parity-plot train overlay)."""
        self._test_preds = []
        self._test_labels = []

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=self.lr * 0.01,
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}
