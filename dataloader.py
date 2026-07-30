"""Paired-image data loading for HUGS-Net."""

from __future__ import annotations

from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class PairedImageDataset(Dataset):
    """Load matching degraded/clean images from ``input`` and ``gt`` folders."""

    def __init__(
        self,
        split_dir: str | Path,
        *,
        crop_size: int | None = None,
        random_flip: bool = False,
        return_name: bool = False,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.input_dir = self.split_dir / "input"
        self.target_dir = self.split_dir / "gt"
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.return_name = return_name

        if not self.input_dir.is_dir() or not self.target_dir.is_dir():
            raise FileNotFoundError(
                f"Expected '{self.input_dir}' and '{self.target_dir}' directories"
            )

        self.names = sorted(
            path.name
            for path in self.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not self.names:
            raise RuntimeError(f"No supported images found in '{self.input_dir}'")

        missing = [name for name in self.names if not (self.target_dir / name).is_file()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"Missing target images for: {preview}")

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def _pad_to_crop(image: Image.Image, crop_size: int) -> Image.Image:
        width, height = image.size
        pad_right = max(crop_size - width, 0)
        pad_bottom = max(crop_size - height, 0)
        if pad_right or pad_bottom:
            image = TF.pad(image, (0, 0, pad_right, pad_bottom), padding_mode="reflect")
        return image

    def _paired_transform(
        self, degraded: Image.Image, target: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if degraded.size != target.size:
            raise ValueError(
                f"Image pair has different sizes: {degraded.size} vs {target.size}"
            )

        if self.crop_size is not None:
            degraded = self._pad_to_crop(degraded, self.crop_size)
            target = self._pad_to_crop(target, self.crop_size)
            width, height = degraded.size
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
            crop = (top, left, self.crop_size, self.crop_size)
            degraded = TF.crop(degraded, *crop)
            target = TF.crop(target, *crop)

        if self.random_flip and random.random() < 0.5:
            degraded = TF.hflip(degraded)
            target = TF.hflip(target)

        return TF.to_tensor(degraded), TF.to_tensor(target)

    def __getitem__(self, index: int):
        name = self.names[index]
        with Image.open(self.input_dir / name) as image:
            degraded = image.convert("RGB")
        with Image.open(self.target_dir / name) as image:
            target = image.convert("RGB")

        degraded_tensor, target_tensor = self._paired_transform(degraded, target)
        if self.return_name:
            return degraded_tensor, target_tensor, name
        return degraded_tensor, target_tensor


def build_train_loader(
    dataset_dir: str | Path,
    batch_size: int,
    num_workers: int,
    crop_size: int = 128,
) -> DataLoader:
    dataset = PairedImageDataset(
        Path(dataset_dir) / "train",
        crop_size=crop_size,
        random_flip=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_eval_loader(
    dataset_dir: str | Path,
    *,
    split: str = "valid",
    batch_size: int = 1,
    num_workers: int = 0,
    return_name: bool = False,
) -> DataLoader:
    dataset = PairedImageDataset(
        Path(dataset_dir) / split,
        return_name=return_name,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


# Compatibility names used by the original scripts.
train_dataloader = build_train_loader


def valid_dataloader(path, batch_size=1, num_workers=0):
    return build_eval_loader(
        path,
        split="valid",
        batch_size=batch_size,
        num_workers=num_workers,
    )


def test_dataloader(path, batch_size=1, num_workers=0):
    return build_eval_loader(
        path,
        split="valid",
        batch_size=batch_size,
        num_workers=num_workers,
        return_name=True,
    )
