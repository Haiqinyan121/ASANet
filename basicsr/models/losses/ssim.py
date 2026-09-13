from typing import Union

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from basicsr.models.losses.loss_util import reduce_loss

_reduction_modes = ['none', 'mean', 'sum']


def _gaussian_kernel(kernel_size: int,
                     sigma: float,
                     device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords = coords - kernel_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


def ssim(x: torch.Tensor,
         y: torch.Tensor,
         kernel_size: int = 11,
         kernel_sigma: float = 1.5,
         data_range: Union[int, float] = 1.,
         reduction: str = 'mean',
         downsample: bool = True,
         k1: float = 0.01,
         k2: float = 0.03) -> torch.Tensor:
    if reduction not in _reduction_modes:
        raise ValueError(f'Unsupported reduction mode: {reduction}. '
                         f'Supported ones are: {_reduction_modes}')
    if x.shape != y.shape:
        raise ValueError(f'Input shapes must match, but got {x.shape} and {y.shape}.')
    if x.dim() != 4:
        raise ValueError(f'SSIM expects 4D tensors, but got {x.dim()}D input.')
    if kernel_size % 2 != 1:
        raise ValueError(f'Kernel size must be odd, got {kernel_size}.')

    x = x.to(torch.float32) / data_range
    y = y.to(torch.float32) / data_range

    downsample_factor = max(1, round(min(x.shape[-2:]) / 256))
    if downsample and downsample_factor > 1:
        x = F.avg_pool2d(x, kernel_size=downsample_factor)
        y = F.avg_pool2d(y, kernel_size=downsample_factor)

    if x.size(-1) < kernel_size or x.size(-2) < kernel_size:
        raise ValueError(f'Kernel size {kernel_size} is larger than input spatial size {x.shape[-2:]}.')

    channels = x.size(1)
    kernel = _gaussian_kernel(kernel_size, kernel_sigma, x.device, x.dtype)
    kernel = kernel.repeat(channels, 1, 1, 1)

    mu_x = F.conv2d(x, kernel, stride=1, padding=0, groups=channels)
    mu_y = F.conv2d(y, kernel, stride=1, padding=0, groups=channels)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, stride=1, padding=0, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, stride=1, padding=0, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, stride=1, padding=0, groups=channels) - mu_xy

    c1 = k1**2
    c2 = k2**2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12)
    ssim_score = ssim_map.mean(dim=(-1, -2)).mean(dim=1)
    return reduce_loss(ssim_score, reduction)


class SSIMLoss(_Loss):
    def __init__(self,
                 loss_weight=1.0,
                 kernel_size: int = 11,
                 kernel_sigma: float = 1.5,
                 k1: float = 0.01,
                 k2: float = 0.03,
                 downsample: bool = True,
                 reduction: str = 'mean',
                 data_range: Union[int, float] = 1.) -> None:
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        if kernel_size % 2 != 1:
            raise ValueError(f'Kernel size must be odd, got {kernel_size}.')

        self.loss_weight = loss_weight
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.downsample = downsample
        self.reduction = reduction
        self.data_range = data_range

    def forward(self, pred, target, weight=None, **kwargs):
        del weight, kwargs
        score = ssim(
            x=pred,
            y=target,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            data_range=self.data_range,
            reduction=self.reduction,
            downsample=self.downsample,
            k1=self.k1,
            k2=self.k2)
        return self.loss_weight * (torch.ones_like(score) - score)
