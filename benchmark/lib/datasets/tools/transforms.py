#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from PIL import Image
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
from monai.utils import ImageMetaKey as Key
from monai.utils import Method, NumpyPadMode, PytorchPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple
from monai.utils.enums import PostFix, TraceKeys

class Normalize(object):
    """Normalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        mean: (list): the mean of RGB
        std: (list): the std of RGB

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, div_value, mean, std):
        self.div_value = div_value
        self.mean = mean
        self.std =std

    def __call__(self, inputs):
        inputs = inputs.div(self.div_value)
        for t, m, s in zip(inputs, self.mean, self.std):
            t.sub_(m).div_(s)

        return inputs

class DeNormalize(object):
    """DeNormalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        mean: (list): the mean of RGB
        std: (list): the std of RGB

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, div_value, mean, std):
        self.div_value = div_value
        self.mean = mean
        self.std =std

    def __call__(self, inputs):
        result = inputs.clone()
        for i in range(result.size(0)):
            result[i, :, :] = result[i, :, :] * self.std[i] + self.mean[i]

        return result.mul_(self.div_value)


class ToTensor(object):
    """Convert a ``numpy.ndarray or Image`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        inputs (numpy.ndarray or Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    def __call__(self, inputs):
        if isinstance(inputs, Image.Image):
            channels = len(inputs.mode)
            inputs = np.array(inputs)
            inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], channels)
            inputs = torch.from_numpy(inputs.transpose(2, 0, 1))
        else:
            inputs = torch.from_numpy(inputs.transpose(2, 0, 1))

        return inputs.float()


class ToLabel(object):
    def __call__(self, inputs):
        return torch.from_numpy(np.array(inputs)).long()


class ReLabel(object):
    """
      255 indicate the background, relabel 255 to some value.
    """
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        assert isinstance(inputs, torch.LongTensor), 'tensor needs to be LongTensor'

        inputs[inputs == self.olabel] = self.nlabel
        return inputs


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        for t in self.transforms:
            inputs = t(inputs)

        return inputs


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


