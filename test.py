"""Evaluate a trained HUGS-Net checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torchvision.transforms.functional import to_pil_image

from dataloader import build_eval_loader
from models.hugs_net import HUGSNet
from utils import (
    AverageMeter,
    load_model_weights,
    peak_signal_to_noise_ratio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test HUGS-Net")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--output-dir", default="results/hugs_net/test")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    if args.save_images:
        output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = build_eval_loader(
        args.dataset_dir,
        split=args.split,
        batch_size=1,
        num_workers=args.num_workers,
        return_name=True,
    )
    model = HUGSNet().to(device)
    load_model_weights(model, args.checkpoint, device)
    model.eval()

    warmup_sample = next(iter(dataloader))[0].to(device)
    for _ in range(args.warmup):
        model(warmup_sample)
    if device.type == "cuda":
        torch.cuda.synchronize()

    psnr_meter = AverageMeter()
    runtime_meter = AverageMeter()
    for index, (degraded, target, names) in enumerate(dataloader, start=1):
        degraded = degraded.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        restored = model(degraded)[-1].clamp(0.0, 1.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        psnr = peak_signal_to_noise_ratio(restored, target).mean().item()
        psnr_meter.update(psnr)
        runtime_meter.update(elapsed)
        print(
            f"{index:04d}/{len(dataloader):04d} "
            f"PSNR {psnr:.3f} dB  Time {elapsed:.6f} s"
        )

        if args.save_images:
            to_pil_image(restored[0].cpu()).save(output_dir / names[0])

    print(f"Average PSNR: {psnr_meter.average:.3f} dB")
    print(f"Average runtime: {runtime_meter.average:.6f} s/image")


if __name__ == "__main__":
    main()
