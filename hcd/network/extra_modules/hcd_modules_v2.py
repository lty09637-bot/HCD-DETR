import torch
import torch.nn as nn

from ..modules.block import C2f
from ..modules.conv import Conv
from .ddg_unit_2 import DDGUnit
from .hcd_modules import HBCN as _LegacyHBCN
from .hcd_modules import WaveletPool

__all__ = (
    'DDAD',
    'DDGM',
    'HBCN',
)


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def attention(self, x):
        pooled = self.pool(x).view(x.shape[0], 1, x.shape[1])
        weight = self.sigmoid(self.conv(pooled))
        return weight.view(x.shape[0], x.shape[1], 1, 1)

    def forward(self, x):
        return x * self.attention(x)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def attention(self, x):
        statistics = torch.cat(
            (torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]),
            dim=1,
        )
        return self.sigmoid(self.conv(statistics))

    def forward(self, x):
        return x * self.attention(x)


class DDAD(nn.Module):
    """Dynamic detail-aware downsampling with depth-dependent channel allocation."""

    def __init__(self, c1, c2, alpha=0.5):
        super().__init__()
        if c1 < 2:
            raise ValueError('DDAD requires at least two input channels')
        if c2 % 2:
            raise ValueError('DDAD output channels must be even')
        if not 0 < alpha < 1:
            raise ValueError('DDAD alpha must be between 0 and 1')

        self.alpha = float(alpha)
        self.detail_channels = int(round(c1 * self.alpha))
        self.semantic_channels = c1 - self.detail_channels
        if not 0 < self.detail_channels < c1:
            raise ValueError('DDAD alpha must allocate channels to both branches')

        branch_channels = c2 // 2
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention()
        self.detail_branch = nn.Sequential(
            WaveletPool(),
            Conv(self.detail_channels * 4, branch_channels, k=1, s=1),
        )
        self.semantic_branch = Conv(self.semantic_channels, branch_channels, k=3, s=2, p=1)
        self.cross_channel_attention = ChannelAttention(branch_channels)
        self.cross_spatial_attention = SpatialAttention()

    def forward(self, x):
        calibrated = self.spatial_attention(self.channel_attention(x))
        detail, semantic = torch.split(
            calibrated,
            (self.detail_channels, self.semantic_channels),
            dim=1,
        )
        detail = self.detail_branch(detail)
        semantic = self.semantic_branch(semantic)
        detail_weight = self.cross_channel_attention.attention(semantic)
        semantic_weight = self.cross_spatial_attention.attention(detail)
        return torch.cat((detail * detail_weight, semantic * semantic_weight), dim=1)


class DDGM(C2f):
    """C2f-based dual-domain denoising-guided module."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DDGUnit(self.c, self.c) for _ in range(n))


class HBCN(_LegacyHBCN):
    """Hierarchical bidirectional calibration network."""

    def __init__(self, inc_list, channel=256, fusion='hbcnfusion'):
        super().__init__(inc_list, channel, fusion)
        p2_channels = inc_list[0]
        self.p2_to_p3 = DDAD(p2_channels, channel)
        self.p2_to_p4 = DDAD(channel, channel)
        self.p2_to_p5 = DDAD(channel, channel)

    def forward(self, x):
        p2, p3, p4, p5 = x

        p5 = self.p5_proj(p5)
        p5_gate = self.p5_cscp(p5)
        p4 = self.p4_proj(p4)
        p4_calibrated = self.p4_calibrate_add([self.p4_calibrate_mul([p4, p5_gate]), p4])
        p4_gate = self.p4_cscp(p4_calibrated)
        p3 = self.p3_proj(p3)
        p3_calibrated = self.p3_calibrate_add([self.p3_calibrate_mul([p3, p4_gate]), p3])

        p4_topdown = self.p4_topdown_fusion([self.p5_upsample(p5), p4_calibrated])
        p4_topdown = self.p4_topdown_block(p4_topdown)
        p3_topdown = self.p3_topdown_fusion([self.p4_upsample(p4_topdown), p3_calibrated])
        p3_topdown = self.p3_topdown_block(p3_topdown)

        p2_to_p3 = self.p2_to_p3(p2)
        p3_gate = self.p3_gate(p2_to_p3)
        p3_out = self.p3_fusion([p2_to_p3, p3_calibrated, p3_topdown])
        p3_out = self.p3_out_add([self.p3_out_mul([p3_out, p3_gate]), p3_out])
        p3_out = self.p3_out_block(p3_out)

        p3_downsampled = self.p3_downsample(p3_out)
        p2_to_p4 = self.p2_to_p4(p2_to_p3)
        p4_gate = self.p4_gate(p2_to_p4)
        p4_out = self.p4_fusion([p3_downsampled, p4_calibrated, p4_topdown])
        p4_out = self.p4_out_add([self.p4_out_mul([p4_out, p4_gate]), p4_out])
        p4_out = self.p4_out_block(p4_out)

        p4_downsampled = self.p4_downsample(p4_out)
        p2_to_p5 = self.p2_to_p5(p2_to_p4)
        p5_gate = self.p5_gate(p2_to_p5)
        p5_out = self.p5_fusion([p4_downsampled, p5])
        p5_out = self.p5_out_add([self.p5_out_mul([p5_out, p5_gate]), p5_out])
        p5_out = self.p5_out_block(p5_out)

        return [p3_out, p4_out, p5_out]
