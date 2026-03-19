from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:

    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


class SobelEdgeExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.shape[1] == 3:
            x = rgb_to_gray(x)

        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)

        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)

        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)

        mag_min = mag.amin(dim=(2, 3), keepdim=True)
        mag_max = mag.amax(dim=(2, 3), keepdim=True)
        mag = (mag - mag_min) / (mag_max - mag_min + 1e-8)

        return mag


class EdgeAwareCoverLoss(nn.Module):

    def __init__(self, lambda_smooth: float = 2.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.edge_extractor = SobelEdgeExtractor()

    def forward(
            self,
            cover: torch.Tensor,
            stego: torch.Tensor
            ) -> torch.Tensor:
        edge_map = self.edge_extractor(cover)
        smooth_map = 1.0 - edge_map
        weight = 1.0 + self.lambda_smooth * smooth_map

        diff2 = (stego - cover) ** 2
        weight = weight.expand_as(diff2)

        loss = (weight * diff2).mean()
        return loss


class StegoLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, cover, stego, secret, recovered_secret):
        cover_loss = self.mse(stego, cover)
        secret_loss = self.mse(recovered_secret, secret)
        total_loss = self.alpha * cover_loss + (1.0 - self.alpha) * secret_loss
        return total_loss, cover_loss, secret_loss


class StegoLossEdgeAware(nn.Module):
    def __init__(self, alpha: float = 0.6, lambda_smooth: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.cover_loss_fn = EdgeAwareCoverLoss(lambda_smooth=lambda_smooth)
        self.secret_loss_fn = nn.MSELoss()

    def forward(self, cover, stego, secret, recovered_secret):
        cover_loss = self.cover_loss_fn(cover, stego)
        secret_loss = self.secret_loss_fn(recovered_secret, secret)
        total_loss = self.alpha * cover_loss + (1.0 - self.alpha) * secret_loss
        return total_loss, cover_loss, secret_loss
