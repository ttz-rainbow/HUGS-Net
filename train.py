"""Train the complete HUGS-Net model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloader import build_eval_loader, build_train_loader
from models.hugs_net import HUGSNet
from utils import (
    AverageMeter,
    current_learning_rate,
    load_model_weights,
    save_checkpoint,
    seed_everything,
    unwrap_model,
)
from valid import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HUGS-Net")
    parser.add_argument("--dataset-dir", required=True, help="Dataset root")
    parser.add_argument("--output-dir", default="results/hugs_net")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--milestones", type=int, nargs="*", default=[50, 80])
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--frequency-weight", type=float, default=0.1)
    parser.add_argument("--save-frequency", type=int, default=10)
    parser.add_argument("--validation-frequency", type=int, default=1)
    parser.add_argument("--print-frequency", type=int, default=100)
    parser.add_argument("--resume", help="Checkpoint path")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device",
    )
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def frequency_l1(prediction: Tensor, target: Tensor) -> Tensor:
    prediction_fft = torch.fft.fft2(prediction, dim=(-2, -1))
    target_fft = torch.fft.fft2(target, dim=(-2, -1))
    return torch.mean(torch.abs(prediction_fft - target_fft))


def reconstruction_loss(
    outputs: list[Tensor],
    target: Tensor,
    frequency_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Equation (10): spatial L1 plus lambda times multi-scale FFT L1."""
    if len(outputs) != 2:
        raise ValueError(f"Expected two HUGS-Net outputs, got {len(outputs)}")

    half_target = F.interpolate(
        target,
        size=outputs[0].shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    targets = (half_target, target)
    spatial = sum(
        F.l1_loss(prediction, reference)
        for prediction, reference in zip(outputs, targets)
    )
    frequency = sum(
        frequency_l1(prediction, reference)
        for prediction, reference in zip(outputs, targets)
    )
    total = spatial + frequency_weight * frequency
    return total, spatial, frequency


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_train_loader(
        args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
    )
    validation_loader = build_eval_loader(
        args.dataset_dir,
        split="valid",
        batch_size=1,
        num_workers=args.num_workers,
    )

    model: nn.Module = HUGSNet().to(device)
    if args.data_parallel:
        if device.type != "cuda":
            raise RuntimeError("--data-parallel requires CUDA")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
    )

    start_epoch = 1
    best_psnr = float("-inf")
    if args.resume:
        checkpoint = load_model_weights(model, args.resume, device)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_psnr = float(checkpoint.get("best_psnr", best_psnr))

    parameter_count = sum(
        parameter.numel() for parameter in unwrap_model(model).parameters()
    )
    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(validation_loader.dataset)}")
    print(f"Model parameters: {parameter_count / 1e6:.3f}M")

    writer = SummaryWriter(log_dir=output_dir / "tensorboard")
    global_step = (start_epoch - 1) * len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_meter = AverageMeter()
        spatial_meter = AverageMeter()
        frequency_meter = AverageMeter()

        for iteration, (degraded, target) in enumerate(train_loader, start=1):
            degraded = degraded.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(degraded)
            total, spatial, frequency = reconstruction_loss(
                outputs, target, args.frequency_weight
            )
            total.backward()
            optimizer.step()

            batch_size = degraded.shape[0]
            total_meter.update(total.item(), batch_size)
            spatial_meter.update(spatial.item(), batch_size)
            frequency_meter.update(frequency.item(), batch_size)
            global_step += 1

            if iteration % args.print_frequency == 0 or iteration == len(train_loader):
                print(
                    f"Epoch {epoch:03d}/{args.epochs:03d} "
                    f"Iter {iteration:04d}/{len(train_loader):04d} "
                    f"LR {current_learning_rate(optimizer):.3e} "
                    f"Loss {total_meter.average:.5f} "
                    f"Spatial {spatial_meter.average:.5f} "
                    f"FFT {frequency_meter.average:.5f}"
                )

            writer.add_scalar("train/total_loss", total.item(), global_step)
            writer.add_scalar("train/spatial_loss", spatial.item(), global_step)
            writer.add_scalar("train/frequency_loss", frequency.item(), global_step)

        scheduler.step()
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_psnr=best_psnr,
        )

        if epoch % args.save_frequency == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_psnr=best_psnr,
            )

        if epoch % args.validation_frequency == 0:
            validation_psnr = evaluate(model, validation_loader, device)
            writer.add_scalar("validation/psnr", validation_psnr, epoch)
            print(f"Validation PSNR: {validation_psnr:.3f} dB")
            if validation_psnr >= best_psnr:
                best_psnr = validation_psnr
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_psnr=best_psnr,
                )

    save_checkpoint(
        checkpoint_dir / "final.pt",
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        best_psnr=best_psnr,
    )
    writer.close()


if __name__ == "__main__":
    main()
