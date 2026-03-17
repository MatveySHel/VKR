import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)


class Encoder(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(6, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock(base_channels)
        self.res2 = ResidualBlock(base_channels)
        self.res3 = ResidualBlock(base_channels)
        self.res4 = ResidualBlock(base_channels)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)
        x = self.in_conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return self.out_conv(x)


class Decoder(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock(base_channels)
        self.res2 = ResidualBlock(base_channels)
        self.res3 = ResidualBlock(base_channels)
        self.res4 = ResidualBlock(base_channels)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, stego):
        x = self.in_conv(stego)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return self.out_conv(x)


class ResNetStegoNet(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.encoder = Encoder(base_channels)
        self.decoder = Decoder(base_channels)

    def forward(self, cover, secret):
        stego = self.encoder(cover, secret)
        recovered_secret = self.decoder(stego)
        return stego, recovered_secret
