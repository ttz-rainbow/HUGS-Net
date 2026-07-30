"""Shared training and evaluation utilities."""

from __future__ import annotations

import random
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor, nn


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * count
        self.count += count

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)

    def __call__(self, value: float) -> None:
        self.update(value)


class Timer:
    def __init__(self) -> None:
        self.start_time = time.perf_counter()

    def reset(self) -> None:
        self.start_time = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def peak_signal_to_noise_ratio(
    prediction: Tensor, target: Tensor, data_range: float = 1.0
) -> Tensor:
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    maximum = torch.tensor(data_range**2, device=mse.device, dtype=mse.dtype)
    return 10.0 * torch.log10(maximum / mse.clamp_min(1e-12))


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def strip_module_prefix(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }


def load_model_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    strict: bool = True,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = strip_module_prefix(state_dict)
    unwrap_model(model).load_state_dict(state_dict, strict=strict)
    return checkpoint


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    epoch: int | None = None,
    best_psnr: float | None = None,
) -> None:
    state = {"model": unwrap_model(model).state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if best_psnr is not None:
        state["best_psnr"] = best_psnr

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


# Backward-compatible names for small external scripts.
Adder = AverageMeter
check_lr = current_learning_rate
