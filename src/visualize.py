import matplotlib.pyplot as plt
import torch

from .metrics import tensor_mse


def tensor_psnr(x, y, max_val=1.0):
    mse = tensor_mse(x, y)
    if mse == 0:
        return float("inf")
    import math
    return 10 * math.log10((max_val ** 2) / mse)


@torch.no_grad()
def visualize_predictions(model, loader, device, n_samples=2):
    model.eval()
    batch = next(iter(loader))

    cover = batch["cover"].to(device)
    secret = batch["secret"].to(device)
    stego, recovered_secret = model(cover, secret)

    cover = cover.cpu()
    secret = secret.cpu()
    stego = stego.cpu()
    recovered_secret = recovered_secret.cpu()
    residual = torch.abs(stego - cover)

    n = min(n_samples, cover.size(0))

    for i in range(n):
        cover_psnr = tensor_psnr(cover[i], stego[i])
        secret_psnr = tensor_psnr(secret[i], recovered_secret[i])

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        axes[0].imshow(cover[i].permute(1, 2, 0).clamp(0, 1))
        axes[0].set_title("Cover")
        axes[0].axis("off")

        axes[1].imshow(secret[i].permute(1, 2, 0).clamp(0, 1))
        axes[1].set_title("Secret")
        axes[1].axis("off")

        axes[2].imshow(stego[i].permute(1, 2, 0).clamp(0, 1))
        axes[2].set_title(f"Stego\nPSNR={cover_psnr:.2f}")
        axes[2].axis("off")

        axes[3].imshow(recovered_secret[i].permute(1, 2, 0).clamp(0, 1))
        axes[3].set_title(f"Recovered\nPSNR={secret_psnr:.2f}")
        axes[3].axis("off")

        res_img = residual[i].permute(1, 2, 0).clamp(0, 1)
        res_img = (res_img / (res_img.max() + 1e-8)).clamp(0, 1)

        axes[4].imshow(res_img)
        axes[4].set_title("Residual")
        axes[4].axis("off")

        plt.tight_layout()
        plt.show()
