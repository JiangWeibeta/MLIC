import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import subpel_conv3x3, AttentionBlock
from modules.layers.conv import conv1x1, conv3x3, conv, deconv
from modules.layers.res_blk import *


class HyperSynthesis(nn.Module):
    """
    Local Reference
    """
    def __init__(self, M=192, N=192) -> None:
        super().__init__()
        self.M = M
        self.N = N

        self.increase = nn.Sequential(
            conv3x3(N, M),
            nn.GELU(),
            subpel_conv3x3(M, M, 2),
            nn.GELU(),
            conv3x3(M, M * 3 // 2),
            nn.GELU(),
            subpel_conv3x3(M * 3 // 2, M * 3 // 2, 2),
            nn.GELU(),
            conv3x3(M * 3 // 2, M * 2),
        )

    def forward(self, x):
        x = self.increase(x)

        return x


class SynthesisTransform(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    def forward(self, x):
        x = self.synthesis_transform(x)

        return x


class SynthesisTransformEX(nn.Module):
    def __init__(self, N, M, act=nn.GELU) -> None:
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            deconv(N, N),
            AttentionBlock(N),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act),
            deconv(N, N),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            ResidualBottleneck(N, act=act, groups=N * 2),
            deconv(N, 3)
        )

    def forward(self, x):
        x = self.synthesis_transform(x)
        return x


class HyperSynthesisEX(nn.Module):
    def __init__(self, N, M, act=nn.GELU) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            deconv(N, M),
            act(),
            deconv(M, M * 3 // 2),
            act(),
            deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.increase(x)
        return x