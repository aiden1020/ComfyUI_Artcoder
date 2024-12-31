import math

import numpy as np
import torch
from torch import nn

from .image_processor import min_max_normalize


class CenterPixelExtractor(nn.Module):
    def __init__(self, module_size: int):
        super().__init__()
        self.module_size = module_size
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=module_size,
            stride=module_size,
            padding=0,
            bias=None,
            groups=1,
        )
        self._setup_kernel_weights()

    def _setup_kernel_weights(self) -> None:
        module_center = int(self.module_size / 2) + 1
        center_filter = torch.zeros((1, 1, self.module_size, self.module_size))
        center_filter[:, :, module_center, module_center] = 1.0
        self.conv.weight = nn.Parameter(center_filter, requires_grad=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RegionMeanFilter(nn.Module):
    def __init__(self, module_size: int):
        super().__init__()
        self.module_size = module_size
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=module_size,
            stride=module_size,
            padding=0,
            bias=None,
        )
        self._setup_kernel_weights()

    def _setup_kernel_weights(self) -> None:
        module_center = int(self.module_size / 2)
        radius = math.ceil(self.module_size / 6)
        center_filter = torch.zeros((1, 1, self.module_size, self.module_size))
        center_filter[
            :, :,
            module_center-radius : module_center+radius,
            module_center-radius : module_center+radius,
        ] = 1.0

        self.conv.weight = nn.Parameter(
            center_filter / center_filter.sum(),
            requires_grad=False,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class QRCodeErrorExtractor(nn.Module):
    def __init__(
        self,
        module_size: int,
        error_mask_black_thres: float = 70 / 255,
        error_mask_white_thres: float = 180 / 255,
    ):
        super().__init__()
        self.module_size = module_size
        self.center_mean_filter = RegionMeanFilter(module_size)
        self.module_size = module_size
        self.error_mask_black_thres = error_mask_black_thres
        self.error_mask_white_thres = error_mask_white_thres

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_center_mean = self.center_mean_filter(x)
        error_mask = (y == 0) & (x_center_mean > self.error_mask_black_thres) | \
                     (y == 1) & (x_center_mean < self.error_mask_white_thres)
        return error_mask.float()


class SamplingSimulationLayer(nn.Module):
    def __init__(self, module_size: int, sampling_threshold: float = 0.1):
        super().__init__()
        self.module_size = module_size
        self.sampling_threshold = sampling_threshold
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=module_size,
            stride=module_size,
            padding=0,
            bias=False,
        )
        self._setup_kernel_weights()

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size) - size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma**2))
        kernel = g[:, None] * g[None, :]
        return kernel / kernel.sum()

    def _setup_kernel_weights(self, sigma=1.5) -> None:
        filter_2d = self._gaussian_kernel(self.module_size, sigma)
        filter_2d = min_max_normalize(filter_2d)
        filter_2d[filter_2d < self.sampling_threshold] = .0
        gaussian_kernel_init = filter_2d.reshape(1, 1, *filter_2d.shape)
        gaussian_kernel_init = torch.cat(
            [gaussian_kernel_init, gaussian_kernel_init, gaussian_kernel_init],
            dim=1,
        )
        self.conv.weight = nn.Parameter(gaussian_kernel_init, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
