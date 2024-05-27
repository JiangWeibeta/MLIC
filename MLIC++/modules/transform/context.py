import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange
from modules.layers import MLP, build_position_index
from modules.layers import conv, deconv
from utils.ckbd import *


class LocalContext(nn.Module):
    def __init__(self,
                 dim=32,
                 window_size=5,
                 mlp_ratio=2.,
                 num_heads=2,
                 qkv_bias=True,
                 qk_scale=None
                ) -> None:
        super().__init__()
        self.H = -1
        self.W = -1
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=window_size, stride=1, padding=(window_size - 1) // 2)
        self.relative_position_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        trunc_normal_(self.relative_position_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim * 2, dim * 2)
        self.mlp = MLP(in_dim=dim * 2, hidden_dim=int(dim * 2 * mlp_ratio), out_dim=dim * 2)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim * 2)
        self.register_buffer("relative_position_index", build_position_index((window_size, window_size)))
        self.attn_mask = None
        self.fusion = nn.Conv2d(dim, dim * 2, kernel_size=window_size)

    def update_resolution(self, H, W, device, mask=None):
        updated=False
        if self.H != H or self.W != W:
            self.H = H
            self.W = W
            if mask is not None:
                self.attn_mask = mask.to(device)
                updated=True
                return updated
            ckbd = torch.zeros((1, 2, H, W), requires_grad=False)
            # anchor
            ckbd[:, :, 0::2, 1::2] = 1
            ckbd[:, :, 1::2, 0::2] = 1
            qk_windows = self.unfold(ckbd).permute(0, 2, 1)
            qk_windows = qk_windows.view(1, H * W, 2, 1, self.window_size, self.window_size).permute(2, 0, 1, 3, 4, 5)
            q_windows, k_windows = qk_windows[0], qk_windows[1]
            q = q_windows.reshape(1, H * W, 1, self.window_size * self.window_size).permute(0, 1, 3, 2)
            k = k_windows.reshape(1, H * W, 1, self.window_size * self.window_size).permute(0, 1, 3, 2)
            attn_mask = (q @ k.transpose(-2, -1))
            attn_mask = attn_mask.masked_fill(attn_mask == 0., float(-100.0)).masked_fill(attn_mask == 1, float(0.0))
            self.attn_mask = attn_mask[0].to(device).detach()
            updated=True
        return updated

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        self.update_resolution(H, W, x.device)
        # [B, L, C]
        x = x.reshape(B, C, L).permute(0, 2, 1)
        x = self.norm1(x)

        # [3, B, C, H, W]
        qkv = self.qkv_proj(x).reshape(B, H, W, 3, C).permute(3, 0, 4, 1, 2)

        # window partition
        q, k, v = qkv[0], qkv[1], qkv[2]
        qkv = torch.cat([q, k, v], dim=1)
        qkv_windows = self.unfold(qkv).permute(0, 2, 1)
        qkv_windows = qkv_windows.view(B, L, 3, C, self.window_size, self.window_size).permute(2, 0, 1, 3, 4, 5)
        # [B, L, C, window_size, window_size]
        q_windows, k_windows, v_windows = qkv_windows[0], qkv_windows[1], qkv_windows[2]

        # [B, L, num_heads, window_size * window_size, head_dim]
        q = q_windows.reshape(B, L, self.head_dim, self.num_heads, self.window_size * self.window_size).permute(0, 1, 3, 4, 2)
        k = k_windows.reshape(B, L, self.head_dim, self.num_heads, self.window_size * self.window_size).permute(0, 1, 3, 4, 2)
        v = v_windows.reshape(B, L, self.head_dim, self.num_heads, self.window_size * self.window_size).permute(0, 1, 3, 4, 2)

        q = q * self.scale
        # [B, L, num_heads, window_size * window_size, window_size * window_size]
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        # [num_heads, window_size * window_size, window_size * window_size]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0).unsqueeze(1)

        attn = attn + self.attn_mask.unsqueeze(0).unsqueeze(2)

        attn = self.softmax(attn)

        x = (attn @ v).reshape(B, L, self.num_heads, self.window_size, self.window_size, self.head_dim).permute(0, 1, 3, 4, 2, 5)
        x = x.reshape(B * L, self.window_size, self.window_size, C).permute(0, 3, 1, 2)
        x = self.fusion(x).reshape(B, L, C * 2)
        x = self.proj(x)
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 2, 1).reshape(B, C * 2, H, W)
        return x


class ChannelContext(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fushion = nn.Sequential(
            nn.Conv2d(in_dim, 192, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(128, out_dim * 4, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, channel_params):
        """
        Args:
            channel_params(Tensor): [B, C * K, H, W]
        return:
            channel_params(Tensor): [B, C * 2, H, W]
        """
        channel_params = self.fushion(channel_params)

        return channel_params


class ChannelContextEX(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU) -> None:
        super().__init__()
        self.fushion = nn.Sequential(
            nn.Conv2d(in_dim, 224, kernel_size=3, stride=1, padding=1),
            act(),
            nn.Conv2d(224, 128, kernel_size=3, stride=1, padding=1),
            act(),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, channel_params):
        """
        Args:
            channel_params(Tensor): [B, C * K, H, W]
        return:
            channel_params(Tensor): [B, C * 2, H, W]
        """
        channel_params = self.fushion(channel_params)

        return channel_params

class LinearGlobalIntraContext(nn.Module):
    def __init__(
            self,
            dim=32,
            num_heads=2) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.keys = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.queries = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.values = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.reprojection = nn.Conv2d(dim, dim * 2, kernel_size=5, stride=1, padding=2)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1, groups=dim * 4),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 2, kernel_size=1, stride=1)
        )

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1_ac = ckbd_anchor(x1)
        x1_na = ckbd_nonanchor(x1)
        queries = ckbd_nonanchor_sequeeze(self.queries(x1_na)).reshape(B, self.dim, H * W//2)
        keys = ckbd_anchor_sequeeze(self.keys(x1_ac)).reshape(B, self.dim, H * W//2)
        values = ckbd_anchor_sequeeze(self.values(x2)).reshape(B, self.dim, H * W//2)
        head_dim = self.dim // self.num_heads

        attended_values = []
        for i in range(self.num_heads):
            key = F.softmax(keys[:, i * head_dim: (i + 1) * head_dim, :], dim=2)
            query = F.softmax(queries[:, i * head_dim: (i + 1) * head_dim, :], dim=1)
            value = values[:, i * head_dim: (i + 1) * head_dim, :]
            key = ckbd_anchor_unsequeeze(key.reshape(B, head_dim, H, W //2)).reshape(B, head_dim, H * W)
            value = ckbd_anchor_unsequeeze(value.reshape(B, head_dim, H, W //2)).reshape(B, head_dim, H * W)
            query = ckbd_nonanchor_unsequeeze(query.reshape(B, head_dim, H, W //2)).reshape(B, head_dim, H * W)
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(B, head_dim, H, W)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention + self.mlp(attention)

class LinearGlobalInterContext(nn.Module):
    def __init__(
            self,
            dim=32,
            out_dim=64,
            num_heads=2) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.keys = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.queries = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.values = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.reprojection = nn.Conv2d(dim, out_dim * 3 // 2, kernel_size=5, stride=1, padding=2)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim * 3 // 2, out_dim * 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim * 2, kernel_size=3, stride=1, padding=1, groups=out_dim * 2),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, stride=1)
        )
        self.skip = nn.Conv2d(out_dim * 3 // 2, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        B, C, H, W = x1.shape
        queries = self.queries(x1).reshape(B, self.dim, H * W)
        keys = self.keys(x1).reshape(B, self.dim, H * W)
        values = self.values(x1).reshape(B, self.dim, H * W)
        head_dim = self.dim // self.num_heads

        attended_values = []
        for i in range(self.num_heads):
            key = F.softmax(keys[:, i * head_dim: (i + 1) * head_dim, :], dim=2)
            query = F.softmax(queries[:, i * head_dim: (i + 1) * head_dim, :], dim=1)
            value = values[:, i * head_dim: (i + 1) * head_dim, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(B, head_dim, H, W)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return self.skip(attention) + self.mlp(attention)
