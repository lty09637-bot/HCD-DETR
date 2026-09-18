import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv, DSConv

__all__ = ('DDGUnit', 'FE', 'LPGE')


class _SafeBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        values_per_channel = x.numel() // x.shape[1]
        if self.training and values_per_channel == 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        return super().forward(x)


class _FrequencyDenoising(nn.Module):
    def __init__(self, in_channels, out_channels, mask_size=32, gaussian_sigma=0.35):
        super().__init__()
        if mask_size < 2:
            raise ValueError('mask_size must be at least 2')
        if gaussian_sigma <= 0:
            raise ValueError('gaussian_sigma must be positive')

        self.preprocess = DSConv(in_channels, out_channels, 3)
        coordinates = torch.linspace(-1.0, 1.0, mask_size)
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing='ij')
        gaussian = torch.exp(-(grid_x.square() + grid_y.square()) / (2.0 * gaussian_sigma ** 2))
        gaussian = gaussian.view(1, 1, mask_size, mask_size)
        self.real_mask = nn.Parameter(gaussian.clone())
        self.imag_mask = nn.Parameter(gaussian.clone())

    @staticmethod
    def _channel_guidance(x):
        return x * torch.sigmoid(F.adaptive_avg_pool2d(x, 1))

    @staticmethod
    def _resize_mask(mask, size, dtype, device):
        resized = F.interpolate(mask, size=size, mode='bilinear', align_corners=False)
        return resized.to(device=device, dtype=dtype)

    def forward(self, x):
        guided = self._channel_guidance(self.preprocess(x))
        fft_input = guided.float() if guided.dtype in (torch.float16, torch.bfloat16) else guided
        spectrum = torch.fft.fftshift(torch.fft.fft2(fft_input, norm='backward'), dim=(-2, -1))
        real_mask = self._resize_mask(self.real_mask, guided.shape[-2:], spectrum.real.dtype, spectrum.device)
        imag_mask = self._resize_mask(self.imag_mask, guided.shape[-2:], spectrum.imag.dtype, spectrum.device)
        filtered = torch.complex(spectrum.real * real_mask, spectrum.imag * imag_mask)
        filtered = torch.fft.ifftshift(filtered, dim=(-2, -1))
        spatial = torch.fft.ifft2(filtered, norm='backward').real
        return guided + spatial.to(guided.dtype)


class LPGE(nn.Module):
    """Local prior-guided enhancement branch described in the paper."""

    def __init__(self, output_dim, patch_size):
        super().__init__()
        if output_dim < 2:
            raise ValueError('output_dim must be at least 2')
        if patch_size < 1:
            raise ValueError('patch_size must be positive')

        patch_area = patch_size * patch_size
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp = nn.Sequential(
            nn.Linear(patch_area, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.Linear(output_dim // 2, output_dim),
        )
        self.query = nn.Parameter(torch.randn(output_dim))
        self.transform = nn.Parameter(torch.eye(output_dim))
        self.projection = nn.Conv2d(output_dim, output_dim, kernel_size=1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        patch_size = self.patch_size
        pad_height = (-height) % patch_size
        pad_width = (-width) % patch_size
        padded = F.pad(x, (0, pad_width, 0, pad_height))
        padded_height, padded_width = padded.shape[-2:]

        patches = F.unfold(padded, kernel_size=patch_size, stride=patch_size)
        patch_count = patches.shape[-1]
        patches = patches.view(batch, channels, patch_size * patch_size, patch_count)
        patches = patches.permute(0, 3, 2, 1)
        local = patches.mean(dim=-1)
        local = local * torch.sigmoid(local)

        local = self.mlp(local)
        local = local * F.softmax(local, dim=-1)
        query = self.query.view(1, 1, -1)
        object_mask = F.cosine_similarity(local, query, dim=-1).clamp(0, 1).unsqueeze(-1)
        local = (local * object_mask) @ self.transform

        local = local.view(batch, padded_height // patch_size, padded_width // patch_size, self.output_dim)
        local = local.permute(0, 3, 1, 2)
        local = F.interpolate(local, size=(height, width), mode='bilinear', align_corners=False)
        return self.projection(local)


class FE(nn.Module):
    """Channel and spatial feature enhancement used at the end of DDGUnit."""

    def __init__(self, channels):
        super().__init__()
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _SafeBatchNorm2d(channels),
            nn.SiLU(),
        )
        self.channel_sigmoid = nn.Sigmoid()
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=False),
            _SafeBatchNorm2d(1),
            nn.SiLU(),
        )
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_weight = self.channel_sigmoid(self.channel_projection(self.channel_pool(x)))
        x = x * channel_weight
        spatial_statistics = torch.cat(
            (torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]),
            dim=1,
        )
        spatial_weight = self.spatial_sigmoid(self.spatial_projection(spatial_statistics))
        return x * spatial_weight


class DDGUnit(nn.Module):
    """Dual-domain denoising-guided unit."""

    def __init__(self, in_features, filters, mask_size=32, gaussian_sigma=0.35):
        super().__init__()
        self.frequency_filter = _FrequencyDenoising(
            in_features,
            filters,
            mask_size=mask_size,
            gaussian_sigma=gaussian_sigma,
        )
        self.lpge_s2 = LPGE(filters, patch_size=2)
        self.lpge_s4 = LPGE(filters, patch_size=4)
        self.fusion = Conv(filters * 3, filters, k=1, s=1)
        self.enhancement = FE(filters)

    def forward(self, x):
        frequency = self.frequency_filter(x)
        local_s2 = self.lpge_s2(x)
        local_s4 = self.lpge_s4(x)
        fused = self.fusion(torch.cat((frequency, local_s2, local_s4), dim=1))
        return self.enhancement(fused)
