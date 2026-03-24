from src.model_wavelet import WaveletStegoNet
from src.model_robust import RobustWrapper
from src.losses import StegoLoss
from src.engine_robust import fit_robust
from src.dataset import StegoPairDataset, build_dataloaders
from src.attacks import (
    CutoutAttack,
    GaussianBlurAttack,
    GaussianNoiseAttack,
    JPEGLikeAttack,
    RandomAttackPipeline,
    ResizeAttack,
)
from configs.config import (
    ALPHA,
    BASE_CHANNELS,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    NUM_WORKERS,
    SEED,
)
import torch
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


def load_split(name: str):
    split_file = DATA_DIR / "splits" / f"{name}.json"
    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    train_paths = load_split("train")
    val_paths = load_split("val")
    test_paths = load_split("test")

    train_ds = StegoPairDataset(train_paths, image_size=IMAGE_SIZE, seed=SEED)
    val_ds = StegoPairDataset(val_paths, image_size=IMAGE_SIZE, seed=SEED)
    test_ds = StegoPairDataset(test_paths, image_size=IMAGE_SIZE, seed=SEED)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    base_model = WaveletStegoNet(base_channels=BASE_CHANNELS)

    attack_pipeline = RandomAttackPipeline(
        attacks=[
            GaussianNoiseAttack(std_min=0.005, std_max=0.03, p=1.0),
            GaussianBlurAttack(
                kernel_size=5, sigma_min=0.5, sigma_max=1.2, p=1.0
            ),
            ResizeAttack(
                scale_min=0.6, scale_max=0.9, p=1.0
            ),
            JPEGLikeAttack(
                scale_min=0.7, scale_max=0.95, q_min=32, q_max=96, p=1.0
            ),
            CutoutAttack(n_holes=3, length=32, p=1.0),
        ],
        p_identity=0.2,
    )

    model = RobustWrapper(base_model, attack_pipeline).to(device)

    criterion = StegoLoss(alpha=ALPHA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    fit_robust(
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
