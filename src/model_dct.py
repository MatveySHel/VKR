from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dct import BlockDCT, BlockIDCT


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
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


class DCTDownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, block_size: int = 2):
        super().__init__()

        self.dct = BlockDCT(block_size=block_size)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch * block_size * block_size, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.dct(x)
        x = self.block(x)
        return x


class IDCTUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, block_size: int = 2):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * block_size * block_size, kernel_size=1),
            nn.BatchNorm2d(out_ch * block_size * block_size),
            nn.ReLU(inplace=True),
        )

        self.idct = BlockIDCT(block_size=block_size)

        self.refine = nn.Sequential(
            ConvBNReLU(out_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.idct(x)
        x = self.refine(x)
        return x


class FuseBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class DCTEncoder(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(6, base_channels),
            ConvBNReLU(base_channels, base_channels),
        )

        self.down1 = DCTDownBlock(base_channels, base_channels * 2)
        self.res1 = ResidualBlock(base_channels * 2)

        self.down2 = DCTDownBlock(base_channels * 2, base_channels * 4)
        self.res2 = ResidualBlock(base_channels * 4)

        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

        self.up1 = IDCTUpBlock(base_channels * 4, base_channels * 2)
        self.fuse1 = FuseBlock(base_channels * 4, base_channels * 2)
        self.res3 = ResidualBlock(base_channels * 2)

        self.up2 = IDCTUpBlock(base_channels * 2, base_channels)
        self.fuse2 = FuseBlock(base_channels * 2, base_channels)
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

        s0 = self.stem(x)

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

        stego = self.head(x)
        return stego


class DCTDecoder(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(3, base_channels),
            ConvBNReLU(base_channels, base_channels),
        )

        self.down1 = DCTDownBlock(base_channels, base_channels * 2)
        self.res1 = ResidualBlock(base_channels * 2)

        self.down2 = DCTDownBlock(base_channels * 2, base_channels * 4)
        self.res2 = ResidualBlock(base_channels * 4)

        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

        self.up1 = IDCTUpBlock(base_channels * 4, base_channels * 2)
        self.fuse1 = FuseBlock(base_channels * 4, base_channels * 2)
        self.res3 = ResidualBlock(base_channels * 2)

        self.up2 = IDCTUpBlock(base_channels * 2, base_channels)
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


class DCTStegoNet(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()

        self.encoder = DCTEncoder(base_channels=base_channels)
        self.decoder = DCTDecoder(base_channels=base_channels)

    def forward(self, cover, secret):
        stego = self.encoder(cover, secret)
        recovered_secret = self.decoder(stego)
        return stego, recovered_secret
