import torch.nn as nn


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
