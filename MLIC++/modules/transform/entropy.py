import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers.attention import MLP


class EntropyParameters(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, 320, kernel_size=1, stride=1, padding=0),
            act(),
            nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0),
            act(),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            act(),
            nn.Conv2d(128, out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, params):
        """
        Args:
            params(Tensor): [B, C * K, H, W]
        return:
            gaussian_params(Tensor): [B, C * 2, H, W]
        """
        gaussian_params = self.fusion(params)

        return gaussian_params


class EntropyParametersEX(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 5 // 3, 1),
            act(),
            nn.Conv2d(out_dim * 5 // 3, out_dim * 4 // 3, 1),
            act(),
            nn.Conv2d(out_dim * 4 // 3, out_dim, 1),
        )

    def forward(self, params):
        """
        Args:
            params(Tensor): [B, C * K, H, W]
        return:
            gaussian_params(Tensor): [B, C * 2, H, W]
        """
        gaussian_params = self.fusion(params)

        return gaussian_params


class ChannelWiseEntropyParameters(nn.Module):
    def __init__(self, in_channels=192, out_channels=192):
        super().__init__()
        diff = (in_channels - out_channels) // 3
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels - diff, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels - diff, in_channels - 2 * diff, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels - 2 * diff, out_channels, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
