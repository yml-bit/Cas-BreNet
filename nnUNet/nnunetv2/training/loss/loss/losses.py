#!/usr/env/bin python3.9
from __future__ import annotations
from typing import List, cast
import torch
import numpy as np
from torch import Tensor, einsum
from .utils import simplex, probs2one_hot, one_hot
from .utils import one_hot2hd_dist

from abc import abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
import torch.nn.functional as F
from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction, convert_data_type, ensure_tuple_rep
from enum import Enum
from monai.utils.type_conversion import convert_to_dst_type
from monai.metrics.metric import CumulativeIterationMetric
from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction, ensure_tuple_rep
import torch.nn as nn

class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss

class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
        union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

        divided: Tensor = 1 - 2 * (einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)

        loss = divided.mean()

        return loss

class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

        divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss

class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss

BoundaryLoss = SurfaceLoss

class HausdorffLoss():
    """
    Implementation heavily inspired from https://github.com/JunMa11/SegWithDistMap
    """
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs)
        assert simplex(target)
        assert probs.shape == target.shape

        B, K, *xyz = probs.shape  # type: ignore

        pc = cast(Tensor, probs[:, self.idc, ...].type(torch.float32))
        tc = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
        assert pc.shape == tc.shape == (B, len(self.idc), *xyz)

        target_dm_npy: np.ndarray = np.stack([one_hot2hd_dist(tc[b].cpu().detach().numpy())
                                              for b in range(B)], axis=0)
        assert target_dm_npy.shape == tc.shape == pc.shape
        tdm: Tensor = torch.tensor(target_dm_npy, device=probs.device, dtype=torch.float32)

        pred_segmentation: Tensor = probs2one_hot(probs).cpu().detach()
        pred_dm_npy: np.nparray = np.stack([one_hot2hd_dist(pred_segmentation[b, self.idc, ...].numpy())
                                            for b in range(B)], axis=0)
        assert pred_dm_npy.shape == tc.shape == pc.shape
        pdm: Tensor = torch.tensor(pred_dm_npy, device=probs.device, dtype=torch.float32)

        delta = (pc - tc)**2
        dtm = tdm**2 + pdm**2

        multipled = einsum("bkwh,bkwh->bkwh", delta, dtm)

        loss = multipled.mean()

        return loss

class FocalLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.gamma: float = kwargs["gamma"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        masked_probs: Tensor = probs[:, self.idc, ...]
        log_p: Tensor = (masked_probs + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        w: Tensor = (1 - masked_probs)**self.gamma
        loss = - einsum("bkwh,bkwh,bkwh->", w, mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # Ensure the predictions are in the same dimension as y_true
        y_pred = torch.sigmoid(y_pred)

        # Calculate the Tversky loss
        tp = (y_true * y_pred).sum(dim=(2, 3, 4))
        fn = (y_true * (1 - y_pred)).sum(dim=(2, 3, 4))
        fp = ((1 - y_true) * y_pred).sum(dim=(2, 3, 4))

        tversky_index = tp / (tp + self.alpha * fn + self.beta * fp)

        # Calculate the Focal Tversky loss
        loss = (1 - tversky_index).pow(self.gamma)

        return loss.mean()

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast
# Assert utils
# def uniq(a: Tensor) -> Set:
#     return set(torch.unique(a.cpu()).numpy())
#
# def sset(a: Tensor, sub: Iterable) -> bool:
#     return uniq(a).issubset(sub)
#
# def one_hot(t: Tensor, axis=1) -> bool:
#     return simplex(t, axis) and sset(t, [0, 1])

def compute_mean_error_metrics(y_pred: torch.Tensor, y: torch.Tensor, func: Callable) -> torch.Tensor:
    # reducing in only channel + spatial dimensions (not batch)
    # reduction of batch handled inside __call__() using do_metric_reduction() in respective calling class
    flt = partial(torch.flatten, start_dim=1)
    return torch.mean(flt(func(y - y_pred)), dim=-1, keepdim=True)

class StrEnum(str, Enum):
    """
    Enum subclass that converts its value to a string.

    .. code-block:: python

        from monai.utils import StrEnum

        class Example(StrEnum):
            MODE_A = "A"
            MODE_B = "B"

        assert (list(Example) == ["A", "B"])
        assert Example.MODE_A == "A"
        assert str(Example.MODE_A) == "A"
        assert monai.utils.look_up_option("A", Example) == "A"
    """

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

class KernelType(StrEnum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"

class RegressionMetric(CumulativeIterationMetric):
    """
    Base class for regression metrics.
    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

    """

    def __init__(self, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False) -> None:
        super().__init__()
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f

    def _check_shape(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        if y_pred.shape != y.shape:
            raise ValueError(f"y_pred and y shapes dont match, received y_pred: [{y_pred.shape}] and y: [{y.shape}]")

        # also check if there is atleast one non-batch dimension i.e. num_dims >= 2
        if len(y_pred.shape) < 2:
            raise ValueError("either channel or spatial dimensions required, found only batch dimension")

    @abstractmethod
    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("y_pred and y must be PyTorch Tensor.")
        self._check_shape(y_pred, y)
        return self._compute_metric(y_pred, y)

class SSIMMetric(RegressionMetric):
    r"""
    Computes the Structural Similarity Index Measure (SSIM).

    .. math::
        \operatorname {SSIM}(x,y) =\frac {(2 \mu_x \mu_y + c_1)(2 \sigma_{xy} + c_2)}{((\mu_x^2 + \
                \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    SSIM reference paper:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
        similarity." IEEE transactions on image processing 13.4 (2004): 600-612.

    Args:
        spatial_dims: number of spatial dimensions of the input images.
        data_range: value range of input images. (usually 1.0 or 255)
        kernel_type: type of kernel, can be "gaussian" or "uniform".
        win_size: window size of kernel
        kernel_sigma: standard deviation for Gaussian kernel.
        k1: stability constant used in the luminance denominator
        k2: stability constant used in the contrast denominator
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans)
    """

    def __init__(
        self,
        spatial_dims: int,
        data_range: float = 1.0,
        kernel_type: KernelType | str = KernelType.GAUSSIAN,
        win_size: int | Sequence[int] = 11,
        kernel_sigma: float | Sequence[float] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)

        self.spatial_dims = spatial_dims
        self.data_range = data_range
        self.kernel_type = kernel_type

        if not isinstance(win_size, Sequence):
            win_size = ensure_tuple_rep(win_size, spatial_dims)
        self.kernel_size = win_size

        if not isinstance(kernel_sigma, Sequence):
            kernel_sigma = ensure_tuple_rep(kernel_sigma, spatial_dims)
        self.kernel_sigma = kernel_sigma

        self.k1 = k1
        self.k2 = k2

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].
            y: Reference image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].

        Raises:
            ValueError: when `y_pred` is not a 2D or 3D image.
        """
        dims = y_pred.ndimension()
        if self.spatial_dims == 2 and dims != 4:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width) when using {self.spatial_dims} "
                f"spatial dimensions, got {dims}."
            )

        if self.spatial_dims == 3 and dims != 5:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width, depth) when using {self.spatial_dims}"
                f" spatial dimensions, got {dims}."
            )

        ssim_value_full_image, _ = compute_ssim_and_cs(
            y_pred=y_pred,
            y=y,
            spatial_dims=self.spatial_dims,
            data_range=self.data_range,
            kernel_type=self.kernel_type,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            k1=self.k1,
            k2=self.k2,
        )

        ssim_per_batch: torch.Tensor = ssim_value_full_image.view(ssim_value_full_image.shape[0], -1).mean(
            1, keepdim=True
        )

        return ssim_per_batch

class SSIMLoss(_Loss):
    """
    Compute the loss function based on the Structural Similarity Index Measure (SSIM) Metric.

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    SSIM reference paper:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
        similarity." IEEE transactions on image processing 13.4 (2004): 600-612.
    """

    def __init__(
        self,
        spatial_dims: int,
        data_range: float = 1.0,
        kernel_type: KernelType | str = KernelType.GAUSSIAN,
        win_size: int | Sequence[int] = 11,
        kernel_sigma: float | Sequence[float] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        reduction: LossReduction | str = LossReduction.MEAN,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions of the input images.
            data_range: value range of input images. (usually 1.0 or 255)
            kernel_type: type of kernel, can be "gaussian" or "uniform".
            win_size: window size of kernel
            kernel_sigma: standard deviation for Gaussian kernel.
            k1: stability constant used in the luminance denominator
            k2: stability constant used in the contrast denominator
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.spatial_dims = spatial_dims
        self.data_range = data_range
        self.kernel_type = kernel_type

        if not isinstance(win_size, Sequence):
            win_size = ensure_tuple_rep(win_size, spatial_dims)
        self.kernel_size = win_size

        if not isinstance(kernel_sigma, Sequence):
            kernel_sigma = ensure_tuple_rep(kernel_sigma, spatial_dims)
        self.kernel_sigma = kernel_sigma

        self.k1 = k1
        self.k2 = k2

        self.ssim_metric = SSIMMetric(
            spatial_dims=self.spatial_dims,
            data_range=self.data_range,
            kernel_type=self.kernel_type,
            win_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            k1=self.k1,
            k2=self.k2,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: batch of predicted images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])
            target: batch of target images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])

        Returns:
            1 minus the ssim index (recall this is meant to be a loss function)

        Example:
            .. code-block:: python

                import torch

                # 2D data
                x = torch.ones([1,1,10,10])/2
                y = torch.ones([1,1,10,10])/2
                print(1-SSIMLoss(spatial_dims=2)(x,y))

                # pseudo-3D data
                x = torch.ones([1,5,10,10])/2  # 5 could represent number of slices
                y = torch.ones([1,5,10,10])/2
                print(1-SSIMLoss(spatial_dims=2)(x,y))

                # 3D data
                x = torch.ones([1,1,10,10,10])/2
                y = torch.ones([1,1,10,10,10])/2
                print(1-SSIMLoss(spatial_dims=3)(x,y))
        """
        input=input.argmax(dim=1).unsqueeze(dim=1)
        ssim_value = self.ssim_metric._compute_tensor(input, target).view(-1, 1)
        loss: torch.Tensor = 1 - ssim_value

        if self.reduction == LossReduction.MEAN.value:
            loss = torch.mean(loss)  # the batch average
        elif self.reduction == LossReduction.SUM.value:
            loss = torch.sum(loss)  # sum over the batch

        return loss

def _gaussian_kernel(
    spatial_dims: int, num_channels: int, kernel_size: Sequence[int], kernel_sigma: Sequence[float]
) -> torch.Tensor:
    """Computes 2D or 3D gaussian kernel.

    Args:
        spatial_dims: number of spatial dimensions of the input images.
        num_channels: number of channels in the image
        kernel_size: size of kernel
        kernel_sigma: standard deviation for Gaussian kernel.
    """

    def gaussian_1d(kernel_size: int, sigma: float) -> torch.Tensor:
        """Computes 1D gaussian kernel.

        Args:
            kernel_size: size of the gaussian kernel
            sigma: Standard deviation of the gaussian kernel
        """
        dist = torch.arange(start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1)
        gauss = torch.exp(-torch.pow(dist / sigma, 2) / 2)
        return (gauss / gauss.sum()).unsqueeze(dim=0)

    gaussian_kernel_x = gaussian_1d(kernel_size[0], kernel_sigma[0])
    gaussian_kernel_y = gaussian_1d(kernel_size[1], kernel_sigma[1])
    kernel = torch.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    kernel_dimensions: tuple[int, ...] = (num_channels, 1, kernel_size[0], kernel_size[1])

    if spatial_dims == 3:
        gaussian_kernel_z = gaussian_1d(kernel_size[2], kernel_sigma[2])[None,]
        kernel = torch.mul(
            kernel.unsqueeze(-1).repeat(1, 1, kernel_size[2]),
            gaussian_kernel_z.expand(kernel_size[0], kernel_size[1], kernel_size[2]),
        )
        kernel_dimensions = (num_channels, 1, kernel_size[0], kernel_size[1], kernel_size[2])

    return kernel.expand(kernel_dimensions)


def compute_ssim_and_cs(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    spatial_dims: int,
    kernel_size: Sequence[int],
    kernel_sigma: Sequence[float],
    data_range: float = 1.0,
    kernel_type: KernelType | str = KernelType.GAUSSIAN,
    k1: float = 0.01,
    k2: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Function to compute the Structural Similarity Index Measure (SSIM) and Contrast Sensitivity (CS) for a batch
    of images.

    Args:
        y_pred: batch of predicted images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])
        y: batch of target images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])
        kernel_size: the size of the kernel to use for the SSIM computation.
        kernel_sigma: the standard deviation of the kernel to use for the SSIM computation.
        spatial_dims: number of spatial dimensions of the images (2, 3)
        data_range: the data range of the images.
        kernel_type: the type of kernel to use for the SSIM computation. Can be either "gaussian" or "uniform".
        k1: the first stability constant.
        k2: the second stability constant.

    Returns:
        ssim: the Structural Similarity Index Measure score for the batch of images.
        cs: the Contrast Sensitivity for the batch of images.
    """
    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    y_pred = convert_data_type(y_pred, output_type=torch.Tensor, dtype=torch.float)[0]
    y = convert_data_type(y, output_type=torch.Tensor, dtype=torch.float)[0]

    num_channels = y_pred.size(1)

    if kernel_type == KernelType.GAUSSIAN:
        kernel = _gaussian_kernel(spatial_dims, num_channels, kernel_size, kernel_sigma)
    elif kernel_type == KernelType.UNIFORM:
        kernel = torch.ones((num_channels, 1, *kernel_size)) / torch.prod(torch.tensor(kernel_size))

    kernel = convert_to_dst_type(src=kernel, dst=y_pred)[0]

    c1 = (k1 * data_range) ** 2  # stability constant for luminance
    c2 = (k2 * data_range) ** 2  # stability constant for contrast

    conv_fn = getattr(F, f"conv{spatial_dims}d")
    mu_x = conv_fn(y_pred, kernel, groups=num_channels)
    mu_y = conv_fn(y, kernel, groups=num_channels)
    mu_xx = conv_fn(y_pred * y_pred, kernel, groups=num_channels)
    mu_yy = conv_fn(y * y, kernel, groups=num_channels)
    mu_xy = conv_fn(y_pred * y, kernel, groups=num_channels)

    sigma_x = mu_xx - mu_x * mu_x
    sigma_y = mu_yy - mu_y * mu_y
    sigma_xy = mu_xy - mu_x * mu_y

    contrast_sensitivity = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
    ssim_value_full_image = ((2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)) * contrast_sensitivity

    return ssim_value_full_image, contrast_sensitivity

