#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import, annotations
from __future__ import division
from __future__ import print_function

import contextlib
from copy import deepcopy
from enum import Enum
from itertools import chain
from math import ceil, floor
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from monai.config import IndexSelection, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms import EnsureChannelFirst
from monai.transforms.croppad.array import (
    BorderPad,
    BoundingRect,
    CenterSpatialCrop,
    CropForeground,
    DivisiblePad,
    RandCropByLabelClasses,
    RandCropByPosNegLabel,
    ResizeWithPadOrCrop,
    SpatialCrop,
    SpatialPad,
)
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, Randomizable
from monai.transforms.utils import (
    allow_missing_keys_mode,
    generate_label_classes_crop_centers,
    generate_pos_neg_label_crop_centers,
    is_positive,
    map_binary_to_indices,
    map_classes_to_indices,
    weighted_patch_samples,
)
from monai.utils import ImageMetaKey as Key, deprecated
from monai.utils import Method, NumpyPadMode, PytorchPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple
from monai.utils.enums import PostFix, TraceKeys
PadModeSequence = Union[Sequence[Union[NumpyPadMode, PytorchPadMode, str]], NumpyPadMode, PytorchPadMode, str]
DEFAULT_POST_FIX = PostFix.meta()

from monai.transforms.transform import Randomizable, Transform
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type
import random
import warnings
import torch
from collections.abc import Mapping, Sequence
from monai.data.meta_tensor import MetaTensor
from monai.transforms.utils_pytorch_numpy_unification import (
    any_np_pt,
    ascontiguousarray,
    cumsum,
    isfinite,
    nonzero,
    ravel,
    searchsorted,
    unique,
    unravel_index,
    where,
)


def correct_crop_centers(
    centers: List[int],
    spatial_size: Union[Sequence[int], int],
    label_spatial_shape: Sequence[int],
    allow_smaller: bool = False,
):
    """
    Utility to correct the crop center if the crop size and centers are not compatible with the image size.

    Args:
        centers: pre-computed crop centers of every dim, will correct based on the valid region.
        spatial_size: spatial size of the ROIs to be sampled.
        label_spatial_shape: spatial shape of the original label data to compare with ROI.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    """
    spatial_size = fall_back_tuple(spatial_size, default=label_spatial_shape)
    if any(np.subtract(label_spatial_shape, spatial_size) < 0):
        if not allow_smaller:
            raise ValueError("The size of the proposed random crop ROI is larger than the image size.")
        spatial_size = tuple(min(l, s) for l, s in zip(label_spatial_shape, spatial_size))

    # Select subregion to assure valid roi
    valid_start = np.floor_divide(spatial_size, 2)
    # add 1 for random
    valid_end = np.subtract(label_spatial_shape + np.array(1), spatial_size / np.array(2)).astype(np.uint16)
    # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
    # from being too high
    for i, valid_s in enumerate(valid_start):
        # need this because np.random.randint does not work with same start and end
        if valid_s == valid_end[i]:
            valid_end[i] += 1
    valid_centers = []
    for c, v_s, v_e in zip(centers, valid_start, valid_end):
        center_i = min(max(c, v_s), v_e - 1)
        valid_centers.append(int(center_i))
    return valid_centers


def generate_pos_neg_label_crop_centers(
    spatial_size: Union[Sequence[int], int],
    num_samples: int,
    pos_ratio: float,
    label_spatial_shape: Sequence[int],
    fg_indices: NdarrayOrTensor,
    bg_indices: NdarrayOrTensor,
    rand_state: Optional[np.random.RandomState] = None,
    allow_smaller: bool = False,
) -> List[List[int]]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]

    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices: pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.

    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    centers = []
    fg_indices = np.asarray(fg_indices) if isinstance(fg_indices, Sequence) else fg_indices
    bg_indices = np.asarray(bg_indices) if isinstance(bg_indices, Sequence) else bg_indices
    if len(fg_indices) == 0 and len(bg_indices) == 0:
        raise ValueError("No sampling location available.")

    if len(fg_indices) == 0 or len(bg_indices) == 0:
        warnings.warn(
            f"N foreground {len(fg_indices)}, N  background {len(bg_indices)},"
            "unable to generate class balanced samples."
        )
        pos_ratio = 0 if fg_indices.size == 0 else 1
    #pos_ratio = random.choice([0.5, 0.5, 0.05, 0.05, 0.05,0.005,0.005, 0.005,0.005, 0.005])#pos_ratio = pos / (pos + neg)
    # pos_ratio = random.choice([0.5, 0.5, 0.5,0.5, 0.05, 0.05, 0.05, 0.005, 0.005])
    a=random.uniform(0, 0.25)
    pos_ratio = random.choice([0.5, 0.25, a])
    # print("比例")
    # print(pos_ratio)
    for _ in range(num_samples):
        indices_to_use = fg_indices if rand_state.rand() < pos_ratio else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        idx = indices_to_use[random_int]
        center = unravel_index(idx, label_spatial_shape).tolist()
        # shift center to range of valid centers
        centers.append(correct_crop_centers(center, spatial_size, label_spatial_shape, allow_smaller))

    return centers

class RandCropByPosNegLabel(Randomizable, Transform):
    """
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of arrays for all the cropped images.
    For example, crop two (3 x 3) arrays from (5 x 5) array with pos/neg=1::

        [[[0, 0, 0, 0, 0],
          [0, 1, 2, 1, 0],            [[0, 1, 2],     [[2, 1, 0],
          [0, 1, 3, 0, 0],     -->     [0, 1, 3],      [3, 0, 0],
          [0, 0, 0, 0, 0],             [0, 0, 0]]      [0, 0, 0]]
          [0, 0, 0, 0, 0]]]

    If a dimension of the expected spatial size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than expected size, and the cropped
    results of several images may not have exactly same shape.
    And if the crop ROI is partly out of the image, will automatically adjust the crop center to ensure the
    valid crop ROI.

    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `label` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        label: the label image that is used for finding foreground/background, if None, must set at
            `self.__call__`.  Non-zero indicates foreground, zero indicates background.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image: optional image data to help select valid area, can be same as `img` or another image array.
            if not None, use ``label == 0 & image > image_threshold`` to select the negative
            sample (background) center. So the crop center will only come from the valid image areas.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to determine
            the valid image content areas.
        fg_indices: if provided pre-computed foreground indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        bg_indices: if provided pre-computed background indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        label: Optional[NdarrayOrTensor] = None,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image: Optional[NdarrayOrTensor] = None,
        image_threshold: float = 0.0,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
        allow_smaller: bool = False,
    ) -> None:
        self.spatial_size = spatial_size
        self.label = label
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[int]]] = None
        self.fg_indices = fg_indices
        self.bg_indices = bg_indices
        self.allow_smaller = allow_smaller

    def randomize(
        self,
        label: NdarrayOrTensor,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
        image: Optional[NdarrayOrTensor] = None,
    ) -> None:
        if fg_indices is None or bg_indices is None:
            if self.fg_indices is not None and self.bg_indices is not None:
                fg_indices_ = self.fg_indices
                bg_indices_ = self.bg_indices
            else:
                fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size,
            self.num_samples,
            self.pos_ratio,
            label.shape[1:],
            fg_indices_,
            bg_indices_,
            self.R,
            self.allow_smaller,
        )

    def __call__(
        self,
        img: NdarrayOrTensor,
        label: Optional[NdarrayOrTensor] = None,
        image: Optional[NdarrayOrTensor] = None,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
    ) -> List[NdarrayOrTensor]:
        """
        Args:
            img: input data to crop samples from based on the pos/neg ratio of `label` and `image`.
                Assumes `img` is a channel-first array.
            label: the label image that is used for finding foreground/background, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``label == 0 & image > image_threshold`` to select the negative sample(background) center.
                so the crop center will only exist on valid image area. if None, use `self.image`.
            fg_indices: foreground indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            bg_indices: background indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.

        """
        if label is None:
            label = self.label
        if label is None:
            raise ValueError("label should be provided.")
        if image is None:
            image = self.image

        self.randomize(label, fg_indices, bg_indices, image)
        results: List[NdarrayOrTensor] = []
        if self.centers is not None:
            for center in self.centers:
                roi_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
                cropper = SpatialCrop(roi_center=center, roi_size=roi_size)
                results.append(cropper(img))

        return results


class RandCropByPosNegLabeld(Randomizable, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByPosNegLabel`.
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    Suppose all the expected fields specified by `keys` have same shape,
    and add `patch_index` to the corresponding metadata.
    And will return a list of dictionaries for all the cropped images.

    If a dimension of the expected spatial size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected size,
    and the cropped results of several images may not have exactly the same shape.
    And if the crop ROI is partly out of the image, will automatically adjust the crop center
    to ensure the valid crop ROI.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `data[label_key]` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.
        fg_indices_key: if provided pre-computed foreground indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        bg_indices_key: if provided pre-computed background indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
            used to add `patch_index` to the meta dict.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_keys is None, use `key_{postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            used to add `patch_index` to the meta dict.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    backend = RandCropByPosNegLabel.backend

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.centers: Optional[List[List[int]]] = None
        self.allow_smaller = allow_smaller

    def randomize(
        self,
        label: NdarrayOrTensor,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
        image: Optional[NdarrayOrTensor] = None,
    ) -> None:
        if fg_indices is None or bg_indices is None:
            fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size,
            self.num_samples,
            self.pos_ratio,
            label.shape[1:],
            fg_indices_,
            bg_indices_,
            self.R,
            self.allow_smaller,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        fg_indices = d.pop(self.fg_indices_key, None) if self.fg_indices_key is not None else None
        bg_indices = d.pop(self.bg_indices_key, None) if self.bg_indices_key is not None else None

        self.randomize(label, fg_indices, bg_indices, image)
        if self.centers is None:
            raise ValueError("no available ROI centers to crop.")

        # initialize returned list with shallow copy to preserve key ordering
        results: List[Dict[Hashable, NdarrayOrTensor]] = [dict(d) for _ in range(self.num_samples)]

        for i, center in enumerate(self.centers):
            # fill in the extra keys with unmodified data
            for key in set(d.keys()).difference(set(self.keys)):
                results[i][key] = deepcopy(d[key])
            for key in self.key_iterator(d):
                img = d[key]
                orig_size = img.shape[1:]
                roi_size = fall_back_tuple(self.spatial_size, default=orig_size)
                cropper = SpatialCrop(roi_center=tuple(center), roi_size=roi_size)
                results[i][key] = cropper(img)
                self.push_transform(results[i], key, extra_info={"center": center}, orig_size=orig_size)
            # add `patch_index` to the metadata
            for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][Key.PATCH_INDEX] = i  # type: ignore

        return results

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[TraceKeys.ORIG_SIZE])
            current_size = np.asarray(d[key].shape[1:])
            center = transform[TraceKeys.EXTRA_INFO]["center"]
            roi_size = fall_back_tuple(self.spatial_size, default=orig_size)
            cropper = SpatialCrop(roi_center=tuple(center), roi_size=roi_size)  # type: ignore
            # get required pad to start and end
            pad_to_start = np.array([s.indices(o)[0] for s, o in zip(cropper.slices, orig_size)])
            pad_to_end = orig_size - current_size - pad_to_start
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            inverse_transform = BorderPad(pad)
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d
