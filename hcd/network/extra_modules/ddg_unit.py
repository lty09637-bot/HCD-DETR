import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import Conv
from .cfo import CFO

__all__ = ('ddgUnit',)


class _SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x


class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


class _ChannelAttention(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super().__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x


class ddgUnit(nn.Module):
    """Core local-global and frequency-enhanced unit used by DDGM."""

    def __init__(self, in_features, filters) -> None:
        super().__init__()

        self.skip = Conv(in_features, filters, act=False)
        # self.c1 = nn.Sequential(
        #     DSConv(filters, filters, 3),
        #     nn.BatchNorm2d(filters),
        #     nn.SiLU()
        # )
        # self.c2 = nn.Sequential(
        #     DSConv(filters, filters, 3),
        #     nn.BatchNorm2d(filters),
        #     nn.SiLU()
        # )
        # self.c3 = nn.Sequential(
        #     DSConv(filters, filters, 3),
        #     nn.BatchNorm2d(filters),
        #     nn.SiLU()
        # )
        self.c1 = Conv(filters, filters, 3)
        self.c2 = Conv(filters, filters, 3)
        self.c3 = Conv(filters, filters, 3)
        self.c4= Conv(6*filters, filters,1)
        self.sa = _SpatialAttention()
        # self.cn = FCA(filters, reso)
        self.cn = _ChannelAttention(filters)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)
        self.drop = nn.Dropout2d(0.1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.silu = nn.SiLU()
        self.cfo = CFO(filters)

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x_skip)
        x2 = self.c2(x1)
        # x3 = self.c3(x2)
        x3 = self.c3(self.cfo(x2))

        # x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        # x = torch.cat((x1, x2, x3, x_skip, x_lga2 , x_lga4), dim=1)
        x = torch.cat((x1, x2, x3, x_skip, x_lga2, x_lga4), dim=1)
        x = self.c4(x)
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.silu(x)
        return x
