import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from configs.config import (
    DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    BASE_CHANNELS, LEARNING_RATE, EPOCHS, ALPHA, SEED, CHECKPOINT_DIR
)
from src.dataset import StegoPairDataset, build_dataloaders
from src.model_wavelet_v1 import WaveletStegoNetV2
from src.losses import StegoLoss
from src.engine import fit


def load_split(name: str):
    split_file = DATA_DIR / "splits" / f"{name}.json"
    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    train_paths = load_split("train")
    val_paths = load_split("val")
    test_paths = load_split("test")

    print(f"Train: {len(train_paths)}")
    print(f"Val:   {len(val_paths)}")
    print(f"Test:  {len(test_paths)}")

    train_ds = StegoPairDataset(train_paths, image_size=IMAGE_SIZE, seed=SEED)
    val_ds = StegoPairDataset(val_paths, image_size=IMAGE_SIZE, seed=SEED)
    test_ds = StegoPairDataset(test_paths, image_size=IMAGE_SIZE, seed=SEED)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = WaveletStegoNetV2(base_channels=BASE_CHANNELS).to(device)
    criterion = StegoLoss(alpha=ALPHA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=EPOCHS,
        ckpt_dir=CHECKPOINT_DIR,
    )


if __name__ == "__main__":
    main()
