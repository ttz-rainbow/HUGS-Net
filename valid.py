"""Validation loop for HUGS-Net."""

from __future__ import annotations

import torch
from torch import nn

from dataloader import build_eval_loader
from utils import AverageMeter, peak_signal_to_noise_ratio


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    psnr_meter = AverageMeter()

    for degraded, target in dataloader:
        degraded = degraded.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        restored = model(degraded)[-1].clamp(0.0, 1.0)
        batch_psnr = peak_signal_to_noise_ratio(restored, target)
        psnr_meter.update(batch_psnr.mean().item(), degraded.shape[0])

    model.train(was_training)
    return psnr_meter.average


def val(model, args, _epoch=None) -> float:
    """Compatibility wrapper around the original ``val`` entry point."""
    device = next(model.parameters()).device
    dataloader = build_eval_loader(
        args.dataset_dir,
        split="valid",
        batch_size=1,
        num_workers=args.num_workers,
    )
    return evaluate(model, dataloader, device)
