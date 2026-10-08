"""Shared fixtures for the Lightning integration tests."""

from __future__ import annotations

import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, TensorDataset


class TinyModel(nn.Module):
    """Two-linear model with a nonzero-gradient CE loss."""

    def __init__(self, d: int = 8, vocab: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, vocab))

    def forward(self, x):
        return self.net(x)


def dataloader(n: int = 16, d: int = 8, batch_size: int = 4) -> DataLoader:
    """Random CE dataloader (avoids the lightning.demos import)."""
    x = torch.randn(n, d)
    y = torch.randint(0, 4, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def trainer(max_steps: int) -> Trainer:
    """A CPU Trainer with logging and checkpointing off."""
    return Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )


__all__ = ["TinyModel", "dataloader", "trainer"]
