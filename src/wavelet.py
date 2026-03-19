from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT(nn.Module):
    """
    2D Haar Discrete Wavelet Transform.

    Input:
        x: [B, C, H, W]
    Output:
        y: [B, 4C, H/2, W/2]

    Channel order per original channel:
        [LL, LH, HL, HH]
    """

    def __init__(self):
        super().__init__()

        s = 1.0 / math.sqrt(2.0)
        lo = torch.tensor([s, s], dtype=torch.float32)
        hi = torch.tensor([s, -s], dtype=torch.float32)

        ll = torch.outer(lo, lo)
        lh = torch.outer(lo, hi)
        hl = torch.outer(hi, lo)
        hh = torch.outer(hi, hi)

        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"H and W must be even, got {(h, w)}")

        weight = self.kernel.repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight=weight, stride=2, padding=0, groups=c)
        return y


class HaarIDWT(nn.Module):

    def __init__(self):
        super().__init__()

        s = 1.0 / math.sqrt(2.0)
        lo = torch.tensor([s, s], dtype=torch.float32)
        hi = torch.tensor([s, -s], dtype=torch.float32)

        ll = torch.outer(lo, lo)
        lh = torch.outer(lo, hi)
        hl = torch.outer(hi, lo)
        hh = torch.outer(hi, hi)

        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("kernel", kernel)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        b, c4, h, w = y.shape
        if c4 % 4 != 0:
            raise ValueError(
                f"Number of channels must be divisible by 4, got {c4}"
            )

        c = c4 // 4
        weight = self.kernel.repeat(c, 1, 1, 1)

        x = F.conv_transpose2d(y, weight=weight, stride=2, padding=0, groups=c)
        return x
