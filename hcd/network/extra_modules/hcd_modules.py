import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import C2f, RepC3
from ..modules.conv import Conv
from .attention import CSCP, DEP
from .ddg_unit import ddgUnit

__all__ = (
    'Add',
    'Multiply',
    'Fusion',
    'HBCN',
    'DDGM',
    'DDAD',
    'WaveletPool',
    'RepBlock',
)


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(torch.stack(x, dim=0), dim=0)


class Multiply(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x[0] * x[1]


class WaveletPool(nn.Module):
    def __init__(self):
        super(WaveletPool, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                            hl[None,::-1,::-1], hh[None,::-1,::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y


class DDAD(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super().__init__()
        self.c = c2 // 2  # 每个分支的输出通道数
        c1_half = c1 // 2 # 每个分支的输入通道数（在avg_pool和chunk之后）
                # --- 注意力机制 (前置) ---
        self.channel_attention = CSCP(c1) # 注意：这里使用 c1 作为输入通道！
        self.spatial_attention = DEP()


        # 分支 1：使用步长卷积进行空间下采样
        self.cv1 = Conv(c1_half, self.c, k=3, s=2, p=1) # 明确指定 k,s,p 可能更好

        # 分支 2：使用 WaveletPool 模块 + 1x1卷积进行通道调整
        self.wavelet_branch = nn.Sequential(
            WaveletPool(),  # WaveletPool 的输出通道是其输入通道的4倍
                            # 输入是 c1_half，所以 WaveletPool 输出是 c1_half * 4
            Conv(c1_half * 4, self.c, k=1, s=1) # 1x1卷积将通道从 c1_half * 4 调整到 self.c
        )

    def forward(self, x):
        # 0. 先应用双重注意力 (Channel -> Spatial)
        channel_gate = self.channel_attention(x)
        x_c = x * channel_gate # 应用通道注意力
        spatial_gate = self.spatial_attention(x_c)
        x_cs = x_c * spatial_gate # 应用空间注意力
        # PyTorch F.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
        # x_cs = nn.functional.avg_pool2d(x, 2, 1, 0, False, True) # 原始代码是 (x, 2, 1, 0, False, True)

        # 将通道拆分成两半
        # 注意：如果上面的avg_pool2d改变了空间维度，x1和x2的空间维度也会相应改变
        x1, x2 = x_cs.chunk(2, 1) # x1 和 x2 各有 c1_half 通道
        # print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")
        # 处理分支 1
        y1 = self.cv1(x1) # 下采样到 H/2, W/2 (相对于x1的尺寸)

        # 处理分支 2 (使用 WaveletPool 分支)
        y2 = self.wavelet_branch(x2) # WaveletPool内部下采样到 H/2, W/2 (相对于x2的尺寸)
        # print(f"y1 shape: {y1.shape}, y2 shape: {y2.shape}")

        # 拼接两个分支的输出
        return torch.cat((y1, y2), 1)


class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='hbcnfusion', tau=4) -> None:
        super().__init__()

        assert fusion == 'hbcnfusion'
        self.fusion = fusion
        self.tau = tau
        self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
        self.epsilon_scale = nn.Parameter(torch.tensor(4, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-4

    def _compute_dynamic_weight(self):
        abs_w = torch.abs(self.fusion_weight)
        exp_w = torch.exp(abs_w)
        softmax_part = (torch.sgn(self.fusion_weight) * (exp_w / (torch.sum(exp_w) + self.eps)) * self.epsilon_scale)
        final_weight = torch.where(abs_w <= self.tau, self.fusion_weight, softmax_part)
        return final_weight

    def forward(self, x):
        fusion_weight = self._compute_dynamic_weight()
        return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)


class DDGM(C2f):
    """C2f-based DDGM block composed of ddgUnit modules."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(ddgUnit(self.c, self.c) for _ in range(n))


class RepBlock(RepC3):
    """Paper-facing name for the retained RepC3 refinement block."""

    pass


class HBCN(nn.Module):
    """Hierarchical bidirectional calibration neck producing P3, P4, and P5 features."""

    def __init__(self, inc_list, channel=256, fusion='hbcnfusion'):
        super().__init__()
        if len(inc_list) != 4:
            raise ValueError(f'HBCN expects four input features, but received {len(inc_list)}')

        p2_channels, p3_channels, p4_channels, p5_channels = inc_list

        # Former YAML layers 12-20: lateral projection and cross-scale calibration.
        self.p5_proj = Conv(p5_channels, channel)
        self.p5_cscp = CSCP(channel)
        self.p4_proj = Conv(p4_channels, channel)
        self.p4_calibrate_mul = Multiply()
        self.p4_calibrate_add = Add()
        self.p4_cscp = CSCP(channel)
        self.p3_proj = Conv(p3_channels, channel)
        self.p3_calibrate_mul = Multiply()
        self.p3_calibrate_add = Add()

        # Former YAML layers 21-26: top-down fusion path.
        self.p5_upsample = nn.Upsample(None, 2, 'nearest')
        self.p4_topdown_fusion = Fusion([channel, channel], fusion)
        self.p4_topdown_block = RepBlock(channel, channel, 3, 0.5)
        self.p4_upsample = nn.Upsample(None, 2, 'nearest')
        self.p3_topdown_fusion = Fusion([channel, channel], fusion)
        self.p3_topdown_block = RepBlock(channel, channel, 3, 0.5)

        # Former YAML layers 27-32: P3 output branch.
        self.p2_to_p3 = DDAD(p2_channels, channel)
        self.p3_gate = DEP()
        self.p3_fusion = Fusion([channel, channel, channel], fusion)
        self.p3_out_mul = Multiply()
        self.p3_out_add = Add()
        self.p3_out_block = RepBlock(channel, channel, 3, 0.5)

        # Former YAML layers 33-39: P4 output branch.
        self.p3_downsample = Conv(channel, channel, 3, 2)
        self.p2_to_p4 = DDAD(channel, channel)
        self.p4_gate = DEP()
        self.p4_fusion = Fusion([channel, channel, channel], fusion)
        self.p4_out_mul = Multiply()
        self.p4_out_add = Add()
        self.p4_out_block = RepBlock(channel, channel, 3, 0.5)

        # Former YAML layers 40-46: P5 output branch.
        self.p4_downsample = Conv(channel, channel, 3, 2)
        self.p2_to_p5 = DDAD(channel, channel)
        self.p5_gate = DEP()
        self.p5_fusion = Fusion([channel, channel], fusion)
        self.p5_out_mul = Multiply()
        self.p5_out_add = Add()
        self.p5_out_block = RepBlock(channel, channel, 3, 0.5)

    def forward(self, x):
        p2, p3, p4, p5 = x

        p5 = self.p5_proj(p5)
        p5_gate = self.p5_cscp(p5)
        p4 = self.p4_proj(p4)
        p4_calibrated = self.p4_calibrate_add([self.p4_calibrate_mul([p4, p5_gate]), p4])
        p4_gate = self.p4_cscp(p4)
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
