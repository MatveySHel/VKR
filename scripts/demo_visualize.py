import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from configs.config import (
    RAW_DIR, IMAGE_SIZE, MIN_SIZE, MAX_ASPECT_RATIO,
    TRAIN_RATIO, VAL_RATIO, BATCH_SIZE, NUM_WORKERS,
    BASE_CHANNELS, SEED, CHECKPOINT_DIR
)
from src.dataset import collect_valid_images, StegoPairDataset, build_splits, build_dataloaders
from src.model import ResNetStegoNet
from src.visualize import visualize_predictions


def main():
    valid_images = collect_valid_images(
        RAW_DIR,
        min_size=MIN_SIZE,
        max_aspect_ratio=MAX_ASPECT_RATIO,
    )

    dataset = StegoPairDataset(valid_images, image_size=IMAGE_SIZE, seed=SEED)
    train_ds, val_ds, test_ds = build_splits(dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED)
    _, _, test_loader = build_dataloaders(train_ds, val_ds, test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetStegoNet(base_channels=BASE_CHANNELS).to(device)

    checkpoint = torch.load(CHECKPOINT_DIR / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    visualize_predictions(model, test_loader, device, n_samples=3)


if __name__ == "__main__":
    main()
