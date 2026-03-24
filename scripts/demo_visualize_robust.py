import torch
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import (
    BASE_CHANNELS,
    BATCH_SIZE,
    DATA_DIR,
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
from src.model_wavelet import WaveletStegoNet
from src.attacks import (
    CutoutAttack,
    GaussianBlurAttack,
    GaussianNoiseAttack,
    IdentityAttack,
    JPEGLikeAttack,
    ResizeAttack,
)
from src.metrics import tensor_mse
from src.dataset import (
    StegoPairDataset, build_dataloaders, build_splits, collect_valid_images
)


def load_split(name: str):
    split_file = DATA_DIR / "splits" / f"{name}.json"
    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


def tensor_psnr(x, y, max_val=1.0):
    mse = tensor_mse(x, y)
    if mse == 0:
        return float("inf")
    import math

    return 10 * math.log10((max_val**2) / mse)


@torch.no_grad()
def visualize_robust_predictions(
    base_model, loader, device, attack, attack_name: str, n_samples=2
):
    base_model.eval()
    batch = next(iter(loader))

    cover = batch["cover"].to(device)
    secret = batch["secret"].to(device)

    stego = base_model.encoder(cover, secret)
    attacked_stego = attack(stego)
    recovered_secret = base_model.decoder(attacked_stego)

    cover = cover.cpu()
    secret = secret.cpu()
    stego = stego.cpu()
    attacked_stego = attacked_stego.cpu()
    recovered_secret = recovered_secret.cpu()

    n = min(n_samples, cover.size(0))

    for i in range(n):
        cover_psnr = tensor_psnr(cover[i], stego[i])
        attack_psnr = tensor_psnr(stego[i], attacked_stego[i])
        secret_psnr = tensor_psnr(secret[i], recovered_secret[i])

        fig, axes = plt.subplots(1, 5, figsize=(22, 4))

        axes[0].imshow(cover[i].permute(1, 2, 0).clamp(0, 1))
        axes[0].set_title("Cover")
        axes[0].axis("off")

        axes[1].imshow(secret[i].permute(1, 2, 0).clamp(0, 1))
        axes[1].set_title("Secret")
        axes[1].axis("off")

        axes[2].imshow(stego[i].permute(1, 2, 0).clamp(0, 1))
        axes[2].set_title(f"Stego\nPSNR={cover_psnr:.2f}")
        axes[2].axis("off")

        axes[3].imshow(attacked_stego[i].permute(1, 2, 0).clamp(0, 1))
        axes[3].set_title(f"Stego + {attack_name}\nPSNR={attack_psnr:.2f}")
        axes[3].axis("off")

        axes[4].imshow(recovered_secret[i].permute(1, 2, 0).clamp(0, 1))
        axes[4].set_title(f"Recovered\nPSNR={secret_psnr:.2f}")
        axes[4].axis("off")

        plt.tight_layout()
        plt.show()


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
        test_ds,
        test_ds,
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    base_model = WaveletStegoNet(base_channels=BASE_CHANNELS).to(device)

    checkpoint = torch.load(
        CHECKPOINT_DIR / "best.pt",
        map_location=device,
    )

    state_dict = checkpoint["model_state_dict"]
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("stego_model."):
            cleaned_state_dict[k.replace("stego_model.", "")] = v

    base_model.load_state_dict(cleaned_state_dict)

    attack_name = "cutout"

    attacks = {
        "clean": IdentityAttack().to(device),
        "noise": GaussianNoiseAttack(
            std_min=0.01, std_max=0.03, p=1.0
        ).to(device),
        "blur": GaussianBlurAttack(
            kernel_size=5, sigma_min=0.8, sigma_max=1.2, p=1.0
        ).to(device),
        "resize": ResizeAttack(
            scale_min=0.6, scale_max=0.85, p=1.0
        ).to(device),
        "jpeg_like": JPEGLikeAttack(
            scale_min=0.7, scale_max=0.95, q_min=24, q_max=80, p=1.0
        ).to(device),
        "cutout": CutoutAttack(n_holes=1, length=16, p=1.0).to(device),
    }

    attack = attacks[attack_name]

    visualize_robust_predictions(
        base_model=base_model,
        loader=test_loader,
        device=device,
        attack=attack,
        attack_name=attack_name,
        n_samples=3,
    )


if __name__ == "__main__":
    main()
