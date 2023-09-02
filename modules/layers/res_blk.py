import torch
import torch.nn as nn
from torch import Tensor
from compressai.layers import GDN, subpel_conv3x3
from modules.layers.conv import conv1x1, conv3x3


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.GELU(),
                    conv3x3(N // 2, N // 2),
                    nn.GELU(),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.GELU()

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.act = nn.GELU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.act = nn.GELU()
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.act(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.act = nn.GELU()
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class ResidualBottleneck(nn.Module):
    def __init__(self, N=192, act=nn.GELU, groups=1) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            conv1x1(N, N // 2),
            act(),
            nn.Conv2d(N // 2, N // 2, kernel_size=3, stride=1, padding=1),
            act(),
            conv1x1(N // 2, N)
        )

    def forward(self, x):
        out = x + self.branch(x)

        return out
