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


class SynthesisTransformOld(nn.Module):
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


class SynthesisTransform(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.synthesis_transform = nn.Sequential(
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, N, 2),
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
