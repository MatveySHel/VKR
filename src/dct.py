from __future__ import annotations

import math
import torch
import torch.nn as nn


def _create_dct_matrix(
        n: int, device=None,
        dtype=torch.float32
        ) -> torch.Tensor:
    mat = torch.zeros((n, n), device=device, dtype=dtype)

    for k in range(n):
        for i in range(n):
            alpha = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
            mat[k, i] = alpha * math.cos(math.pi * (2 * i + 1) * k / (2 * n))

    return mat


class BlockDCT(nn.Module):

    def __init__(self, block_size: int = 2):
        super().__init__()
        self.block_size = block_size

        dct = _create_dct_matrix(block_size)
        self.register_buffer("dct", dct)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = self.block_size

        assert h % n == 0 and w % n == 0, (
            "H and W must be divisible by block_size"
        )

        dct = self.dct.to(device=x.device, dtype=x.dtype)

        x = x.view(b, c, h // n, n, w // n, n)

        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()

        y = torch.einsum("ij,bchwjk,kl->bchwil", dct, x, dct.t())

        y = y.permute(0, 1, 4, 5, 2, 3).contiguous()
        y = y.view(b, c * n * n, h // n, w // n)

        return y


class BlockIDCT(nn.Module):
    """
    Inverse block-wise 2D DCT.
    Input:  [B, C * block_size * block_size, H, W]
    Output: [B, C, H*block_size, W*block_size]
    """
    def __init__(self, block_size: int = 2):
        super().__init__()
        self.block_size = block_size

        dct = _create_dct_matrix(block_size)
        self.register_buffer("dct", dct)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, cn, h, w = x.shape
        n = self.block_size

        assert cn % (n * n) == 0, "Channels must be divisible by block_size^2"

        c = cn // (n * n)
        dct = self.dct.to(device=x.device, dtype=x.dtype)

        x = x.view(b, c, n, n, h, w)

        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()

        y = torch.einsum("ij,bchwjk,kl->bchwil", dct.t(), x, dct)

        y = y.permute(0, 1, 2, 4, 3, 5).contiguous()
        y = y.view(b, c, h * n, w * n)

        return y
