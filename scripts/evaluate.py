from src.losses import StegoLoss
from src.engine import validate_one_epoch
from src.dataset import (
    StegoPairDataset, build_dataloaders,
    build_splits, collect_valid_images
)
from src.model_wavelet import WaveletStegoNet
from configs.config import (
    ALPHA,
    BASE_CHANNELS,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    IMAGE_SIZE,
    MAX_ASPECT_RATIO,
    MIN_SIZE,
    NUM_WORKERS,
    RAW_DIR,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
)
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


def main():
    valid_images = collect_valid_images(
        RAW_DIR,
        min_size=MIN_SIZE,
        max_aspect_ratio=MAX_ASPECT_RATIO,
    )

    dataset = StegoPairDataset(valid_images, image_size=IMAGE_SIZE, seed=SEED)
    train_ds, val_ds, test_ds = build_splits(
        dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED)
    _, _, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WaveletStegoNet(base_channels=BASE_CHANNELS).to(device)

    checkpoint = torch.load(CHECKPOINT_DIR / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = StegoLoss(alpha=ALPHA)
    stats = validate_one_epoch(model, test_loader, criterion, device)

    print("Test results:")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
