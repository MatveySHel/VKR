# src/model_wavelet_v2.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import HaarDWT, HaarIDWT


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


class DWTDownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dwt = HaarDWT()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ConvBNReLU(out_ch, out_ch, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.dwt(x)
        x = self.block(x)
        return x


class IDWTUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1),
            nn.BatchNorm2d(out_ch * 4),
            nn.ReLU(inplace=True),
        )
        self.idwt = HaarIDWT()
        self.refine = nn.Sequential(
            ConvBNReLU(out_ch, out_ch, kernel_size=3, stride=1),
            ConvBNReLU(out_ch, out_ch, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.idwt(x)
        x = self.refine(x)
        return x


class FuseBlock(nn.Module):
    """
    Fuse decoder features with skip features after concatenation.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, kernel_size=3, stride=1),
            ConvBNReLU(out_ch, out_ch, kernel_size=3, stride=1),
        )

    def forward(self, x):
        return self.block(x)


class WaveletEncoderV2(nn.Module):
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

        self.down1 = DWTDownBlock(base_channels, base_channels * 2)      # 256 -> 128
        self.res1 = ResidualBlock(base_channels * 2)

        self.down2 = DWTDownBlock(base_channels * 2, base_channels * 4)  # 128 -> 64
        self.res2 = ResidualBlock(base_channels * 4)

        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

        self.up1 = IDWTUpBlock(base_channels * 4, base_channels * 2)     # 64 -> 128
        self.fuse1 = FuseBlock(base_channels * 4, base_channels * 2)     # concat skip1
        self.res3 = ResidualBlock(base_channels * 2)

        self.up2 = IDWTUpBlock(base_channels * 2, base_channels)         # 128 -> 256
        self.fuse2 = FuseBlock(base_channels * 2, base_channels)         # concat stem skip
        self.res4 = ResidualBlock(base_channels)

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)

        s0 = self.stem(x)        # [B, base, 256,256]

        x1 = self.down1(s0)      # [B, 2base, 128,128]
        s1 = self.res1(x1)       # skip 1

        x2 = self.down2(s1)      # [B, 4base, 64,64]
        x2 = self.res2(x2)

        x = self.bottleneck(x2)

        x = self.up1(x)          # [B, 2base, 128,128]
        x = torch.cat([x, s1], dim=1)
        x = self.fuse1(x)
        x = self.res3(x)

        x = self.up2(x)          # [B, base, 256,256]
        x = torch.cat([x, s0], dim=1)
        x = self.fuse2(x)
        x = self.res4(x)

        stego = self.head(x)
        return stego


class WaveletDecoderV2(nn.Module):
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

        self.down1 = DWTDownBlock(base_channels, base_channels * 2)
        self.res1 = ResidualBlock(base_channels * 2)

        self.down2 = DWTDownBlock(base_channels * 2, base_channels * 4)
        self.res2 = ResidualBlock(base_channels * 4)

        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

        self.up1 = IDWTUpBlock(base_channels * 4, base_channels * 2)
        self.fuse1 = FuseBlock(base_channels * 4, base_channels * 2)
        self.res3 = ResidualBlock(base_channels * 2)

        self.up2 = IDWTUpBlock(base_channels * 2, base_channels)
        self.fuse2 = FuseBlock(base_channels * 2, base_channels)
        self.res4 = ResidualBlock(base_channels)

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, stego):
        s0 = self.stem(stego)

        x1 = self.down1(s0)
        s1 = self.res1(x1)

        x2 = self.down2(s1)
        x2 = self.res2(x2)

        x = self.bottleneck(x2)

        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.fuse1(x)
        x = self.res3(x)

        x = self.up2(x)
        x = torch.cat([x, s0], dim=1)
        x = self.fuse2(x)
        x = self.res4(x)

        recovered_secret = self.head(x)
        return recovered_secret


class WaveletStegoNetV2(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.encoder = WaveletEncoderV2(base_channels=base_channels)
        self.decoder = WaveletDecoderV2(base_channels=base_channels)

    def forward(self, cover, secret):
        stego = self.encoder(cover, secret)
        recovered_secret = self.decoder(stego)
        return stego, recovered_secret
