# src/engine_robust.py

from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from .metrics import mse_to_psnr


def train_one_epoch_robust(model, loader, optimizer, criterion, device):
    model.train()
    stats = defaultdict(float)
    num_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        cover = batch["cover"].to(device)
        secret = batch["secret"].to(device)

        optimizer.zero_grad()

        stego, attacked_stego, recovered_secret = model(cover, secret)
        loss, cover_loss, secret_loss = criterion(
            cover,
            stego,
            secret,
            recovered_secret
        )

        loss.backward()
        optimizer.step()

        stats["loss"] += loss.item()
        stats["cover_loss"] += cover_loss.item()
        stats["secret_loss"] += secret_loss.item()
        num_batches += 1

    for k in stats:
        stats[k] /= num_batches

    stats["cover_psnr"] = mse_to_psnr(stats["cover_loss"])
    stats["secret_psnr"] = mse_to_psnr(stats["secret_loss"])
    return dict(stats)


@torch.no_grad()
def validate_one_epoch_robust(model, loader, criterion, device):
    model.eval()
    stats = defaultdict(float)
    num_batches = 0

    pbar = tqdm(loader, desc="Val", leave=False)

    for batch in pbar:
        cover = batch["cover"].to(device)
        secret = batch["secret"].to(device)

        stego, attacked_stego, recovered_secret = model(cover, secret)
        loss, cover_loss, secret_loss = criterion(
            cover,
            stego,
            secret,
            recovered_secret
        )

        stats["loss"] += loss.item()
        stats["cover_loss"] += cover_loss.item()
        stats["secret_loss"] += secret_loss.item()
        num_batches += 1

    for k in stats:
        stats[k] /= num_batches

    stats["cover_psnr"] = mse_to_psnr(stats["cover_loss"])
    stats["secret_psnr"] = mse_to_psnr(stats["secret_loss"])
    return dict(stats)


def fit_robust(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs=20,
        ckpt_dir="./checkpoints"
        ):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch_robust(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )
        val_stats = validate_one_epoch_robust(
            model,
            val_loader,
            criterion,
            device
        )

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"train_cover_psnr={train_stats['cover_psnr']:.2f} | "
            f"val_cover_psnr={val_stats['cover_psnr']:.2f} | "
            f"train_secret_psnr={train_stats['secret_psnr']:.2f} | "
            f"val_secret_psnr={val_stats['secret_psnr']:.2f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            ckpt_dir / "last.pt",
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                ckpt_dir / "best.pt",
            )

    return history
