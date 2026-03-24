from src.model_wavelet import WaveletStegoNet
from src.metrics import mse_to_psnr
from src.losses import StegoLoss
from src.dataset import StegoPairDataset, build_dataloaders
from src.attacks import (
    CutoutAttack,
    GaussianBlurAttack,
    GaussianNoiseAttack,
    IdentityAttack,
    JPEGLikeAttack,
    ResizeAttack,
)
from configs.config import (
    ALPHA, BASE_CHANNELS, BATCH_SIZE, CHECKPOINT_DIR,
    DATA_DIR, IMAGE_SIZE, NUM_WORKERS, SEED
)
from tqdm import tqdm
import torch
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


def load_split(name: str):
    split_file = DATA_DIR / "splits" / f"{name}.json"
    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def evaluate_with_attack(
    base_model, loader, criterion, device, attack, attack_name: str
):
    base_model.eval()
    stats = defaultdict(float)
    num_batches = 0

    pbar = tqdm(loader, desc=f"Eval [{attack_name}]", leave=False)

    for batch in pbar:
        cover = batch["cover"].to(device)
        secret = batch["secret"].to(device)

        stego = base_model.encoder(cover, secret)
        attacked_stego = attack(stego)
        recovered_secret = base_model.decoder(attacked_stego)

        loss, cover_loss, secret_loss = criterion(
            cover, stego, secret, recovered_secret)

        stats["loss"] += loss.item()
        stats["cover_loss"] += cover_loss.item()
        stats["secret_loss"] += secret_loss.item()
        num_batches += 1

    for k in stats:
        stats[k] /= num_batches

    stats["cover_psnr"] = mse_to_psnr(stats["cover_loss"])
    stats["secret_psnr"] = mse_to_psnr(stats["secret_loss"])

    return dict(stats)


def main():
    test_paths = load_split("test")
    test_ds = StegoPairDataset(test_paths, image_size=IMAGE_SIZE, seed=SEED)
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

    criterion = StegoLoss(alpha=ALPHA)

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
        "cutout": CutoutAttack(n_holes=3, length=32, p=1.0).to(device),
    }

    results = {}

    for attack_name, attack in attacks.items():
        stats = evaluate_with_attack(
            base_model, test_loader, criterion, device, attack, attack_name)
        results[attack_name] = stats

    print("\n=== Robust Evaluation Results ===")
    for attack_name, stats in results.items():
        print(f"\n[{attack_name}]")
        for k, v in stats.items():
            print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
