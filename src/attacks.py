from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityAttack(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GaussianNoiseAttack(nn.Module):
    def __init__(
            self,
            std_min: float = 0.0,
            std_max: float = 0.03,
            p: float = 1.0
            ):
        super().__init__()
        self.std_min = std_min
        self.std_max = std_max
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x

        std = random.uniform(self.std_min, self.std_max)
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0.0, 1.0)


class GaussianBlurAttack(nn.Module):
    def __init__(
            self,
            kernel_size: int = 5,
            sigma_min: float = 0.5,
            sigma_max: float = 1.5,
            p: float = 1.0
            ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p

    def _gaussian_kernel(
            self,
            device: torch.device,
            channels: int,
            sigma: float
            ) -> torch.Tensor:
        k = self.kernel_size
        ax = torch.arange(k, device=device) - (k - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, k, k).repeat(channels, 1, 1, 1)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x

        sigma = random.uniform(self.sigma_min, self.sigma_max)
        b, c, _, _ = x.shape
        kernel = self._gaussian_kernel(x.device, c, sigma)
        pad = self.kernel_size // 2
        x = F.conv2d(x, kernel, padding=pad, groups=c)
        return torch.clamp(x, 0.0, 1.0)


class ResizeAttack(nn.Module):

    def __init__(
            self,
            scale_min: float = 0.5,
            scale_max: float = 0.9,
            mode: str = "bilinear",
            p: float = 1.0
            ):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.mode = mode
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x

        _, _, h, w = x.shape
        scale = random.uniform(self.scale_min, self.scale_max)
        nh = max(16, int(h * scale))
        nw = max(16, int(w * scale))

        y = F.interpolate(
            x, size=(nh, nw), mode=self.mode,
            align_corners=False if self.mode in [
                "bilinear", "bicubic"
            ] else None
        )
        y = F.interpolate(
            y, size=(h, w), mode=self.mode,
            align_corners=False if self.mode in [
                "bilinear", "bicubic"
            ] else None
        )
        return torch.clamp(y, 0.0, 1.0)


class JPEGLikeAttack(nn.Module):

    def __init__(
        self,
        scale_min: float = 0.6,
        scale_max: float = 0.95,
        q_min: int = 32,
        q_max: int = 128,
        p: float = 1.0
    ):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.q_min = q_min
        self.q_max = q_max
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x

        _, _, h, w = x.shape
        scale = random.uniform(self.scale_min, self.scale_max)
        nh = max(16, int(h * scale))
        nw = max(16, int(w * scale))

        y = F.interpolate(
            x,
            size=(nh, nw),
            mode="bilinear",
            align_corners=False
        )
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

        q = random.randint(self.q_min, self.q_max)
        y = torch.round(y * q) / q
        return torch.clamp(y, 0.0, 1.0)


class RandomAttackPipeline(nn.Module):

    def __init__(self, attacks: list[nn.Module], p_identity: float = 0.2):
        super().__init__()
        self.attacks = nn.ModuleList(attacks)
        self.p_identity = p_identity
        self.identity = IdentityAttack()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.attacks) == 0 or random.random() < self.p_identity:
            return self.identity(x)

        attack = random.choice(list(self.attacks))
        return attack(x)
