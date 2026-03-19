import math

import torch


def tensor_mse(x, y):
    return torch.mean((x - y) ** 2).item()


def mse_to_psnr(mse_value: float, max_val: float = 1.0) -> float:
    if mse_value == 0:
        return float("inf")
    return 10.0 * math.log10((max_val**2) / mse_value)
