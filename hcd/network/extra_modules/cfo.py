import math

import torch
import torch.nn as nn

from ..modules.conv import DSConv

__all__ = ('CFO',)


class _FrequencyChannelAttention(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class CFO(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.c1 = nn.Sequential(
            DSConv(dim, dim, 3),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        self.c2 = nn.Sequential(
            DSConv(dim, dim, 3),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        self.c3 = nn.Sequential(
            DSConv(dim, dim, 3),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        # self.conv1 = Conv(dim, dim, 3)
        # self.conv2 = Conv(dim, dim, 3)
        # self.conv3 = Conv(dim, dim, 3)
        self.bn = nn.BatchNorm2d(dim)
        # self.relu = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))
        self.eca = _FrequencyChannelAttention(dim)

    def forward(self, x):
        out = self.c1(x)
        out = self.eca(out)
        identity = out
        out_fft = torch.fft.fft2(out, norm='backward')
        out_real = torch.real(out_fft)
        out_imag = torch.imag(out_fft)
        out_real = self.c2(out_real)
        out_imag = self.c3(out_imag)
        out = torch.complex(out_real, out_imag)
        out = torch.abs(out)
        out = out + identity

        return out
