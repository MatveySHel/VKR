from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError, ImageFile
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_valid_image(
    path: str | Path,
    min_size: int = 128,
    max_aspect_ratio: float = 3.0,
) -> bool:
    path = Path(path)

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size

            if w < min_size or h < min_size:
                return False

            ratio = max(w / h, h / w)
            if ratio > max_aspect_ratio:
                return False

        return True

    except (UnidentifiedImageError, OSError, ValueError):
        return False


def collect_valid_images(
    image_dir: str | Path,
    valid_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    min_size: int = 128,
    max_aspect_ratio: float = 3.0,
) -> list[Path]:
    image_dir = Path(image_dir)

    all_paths = [
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in valid_exts
    ]

    valid_paths = [
        p for p in all_paths
        if is_valid_image(p, min_size=min_size, max_aspect_ratio=max_aspect_ratio)
    ]

    return sorted(valid_paths)


class StegoPairDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str | Path],
        image_size: int = 256,
        seed: int = 42,
        transform: Optional[transforms.Compose] = None,
        max_retries: int = 10,
    ) -> None:
        if len(image_paths) < 2:
            raise ValueError("Need at least 2 valid images")

        self.image_paths = [Path(p) for p in image_paths]
        self.rng = random.Random(seed)
        self.max_retries = max_retries

        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)

    def _sample_secret_idx(self, idx: int) -> int:
        while True:
            j = self.rng.randrange(len(self.image_paths))
            if j != idx:
                return j

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        for _ in range(self.max_retries):
            try:
                cover_path = self.image_paths[idx]
                secret_idx = self._sample_secret_idx(idx)
                secret_path = self.image_paths[secret_idx]

                cover = self._load_image(cover_path)
                secret = self._load_image(secret_path)

                return {
                    "cover": cover,
                    "secret": secret,
                    "cover_path": str(cover_path),
                    "secret_path": str(secret_path),
                }

            except (OSError, UnidentifiedImageError, ValueError):
                idx = self.rng.randrange(len(self.image_paths))

        raise RuntimeError("Too many failed image loading attempts")


def build_splits(dataset: Dataset, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    total = len(dataset)
    train_len = int(total * train_ratio)
    val_len = int(total * val_ratio)
    test_len = total - train_len - val_len

    generator = torch.Generator().manual_seed(seed)

    return random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=generator,
    )


def build_dataloaders(train_ds, val_ds, test_ds, batch_size: int = 4, num_workers: int = 0):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader