from __future__ import annotations

import torch
import torch.nn as nn


class RobustWrapper(nn.Module):
    def __init__(self, stego_model: nn.Module, attack_pipeline: nn.Module):
        super().__init__()
        self.stego_model = stego_model
        self.attack_pipeline = attack_pipeline

    def forward(self, cover: torch.Tensor, secret: torch.Tensor):
        stego = self.stego_model.encoder(cover, secret)
        attacked_stego = self.attack_pipeline(stego)
        recovered_secret = self.stego_model.decoder(attacked_stego)
        return stego, attacked_stego, recovered_secret
