import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)


class DownBlock(nn.Module):
    """
    Reduces spatial size by 2.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, kernel_size=3, stride=2),  # H/2, W/2
            ConvBNReLU(out_ch, out_ch, kernel_size=3, stride=1),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Increases spatial size by 2.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBNReLU(out_ch, out_ch, kernel_size=3, stride=1),
            ConvBNReLU(out_ch, out_ch, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class EncoderV2(nn.Module):
    """
    Input:
        cover  -> [B, 3, 256, 256]
        secret -> [B, 3, 256, 256]
    Output:
        stego  -> [B, 3, 256, 256]
    """
    def __init__(self, base_channels: int = 32):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(6, base_channels),
            ConvBNReLU(base_channels, base_channels),
        )

        self.down1 = DownBlock(base_channels, base_channels * 2)      # 256 -> 128
        self.res1 = ResidualBlock(base_channels * 2)

        self.down2 = DownBlock(base_channels * 2, base_channels * 4)  # 128 -> 64
        self.res2 = ResidualBlock(base_channels * 4)

        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

        self.up1 = UpBlock(base_channels * 4, base_channels * 2)      # 64 -> 128
        self.res3 = ResidualBlock(base_channels * 2)

        self.up2 = UpBlock(base_channels * 2, base_channels)          # 128 -> 256
        self.res4 = ResidualBlock(base_channels)

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)   # [B, 6, 256, 256]

        x = self.stem(x)
        x = self.down1(x)
        x = self.res1(x)

        x = self.down2(x)
        x = self.res2(x)

        x = self.bottleneck(x)

        x = self.up1(x)
        x = self.res3(x)

        x = self.up2(x)
        x = self.res4(x)

        stego = self.head(x)
        return stego


class DecoderV2(nn.Module):
    """
    Input:
        stego             -> [B, 3, 256, 256]
    Output:
        recovered_secret  -> [B, 3, 256, 256]
    """
    def __init__(self, base_channels: int = 32):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(3, base_channels),
            ConvBNReLU(base_channels, base_channels),
        )

        self.down1 = DownBlock(base_channels, base_channels * 2)      # 256 -> 128
        self.res1 = ResidualBlock(base_channels * 2)

        self.down2 = DownBlock(base_channels * 2, base_channels * 4)  # 128 -> 64
        self.res2 = ResidualBlock(base_channels * 4)

        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

        self.up1 = UpBlock(base_channels * 4, base_channels * 2)      # 64 -> 128
        self.res3 = ResidualBlock(base_channels * 2)

        self.up2 = UpBlock(base_channels * 2, base_channels)          # 128 -> 256
        self.res4 = ResidualBlock(base_channels)

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, stego):
        x = self.stem(stego)

        x = self.down1(x)
        x = self.res1(x)

        x = self.down2(x)
        x = self.res2(x)

        x = self.bottleneck(x)

        x = self.up1(x)
        x = self.res3(x)

        x = self.up2(x)
        x = self.res4(x)

        recovered_secret = self.head(x)
        return recovered_secret


class ResNetStegoNetV2(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.encoder = EncoderV2(base_channels=base_channels)
        self.decoder = DecoderV2(base_channels=base_channels)

    def forward(self, cover, secret):
        stego = self.encoder(cover, secret)
        recovered_secret = self.decoder(stego)
        return stego, recovered_secret