import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
# from scipy.ndimage.interpolation import zoom
from scipy.ndimage import zoom
from torch.utils.data import Dataset
import cv2
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import SimpleITK as sitk
from monai.transforms import RandZoomd


def random_rot_flip(image, label):
    # k--> angle
    # i, j: axis
    k = np.random.randint(0, 4)
    axis = random.sample(range(0, 3), 2)
    image = np.rot90(image, k, axes=(axis[0], axis[1]))  # rot along z axis
    label = np.rot90(label, k, axes=(axis[0], axis[1]))

    # axis = np.random.randint(0, 2)
    # image = np.flip(image, axis=axis)
    # label = np.flip(label, axis=axis)
    flip_id = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
    image = np.ascontiguousarray(image[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    label = np.ascontiguousarray(label[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    return image, label

def random_rotate(image, label, min_value):
    angle = np.random.randint(-15, 15)  # -20--20
    rotate_axes = [(0, 1), (1, 2), (0, 2)]
    k = np.random.randint(0, 3)
    image = ndimage.interpolation.rotate(image, angle, axes=rotate_axes[k], reshape=False, order=3, mode='constant',
                                         cval=min_value)
    label = ndimage.interpolation.rotate(label, angle, axes=rotate_axes[k], reshape=False, order=0, mode='constant',
                                         cval=0.0)

    return image, label

# z, y, x     0, 1, 2
def rot_from_y_x(image, label):
    image = np.rot90(image, 2, axes=(1, 2))  # rot along z axis
    label = np.rot90(label, 2, axes=(1, 2))

    return image, label

def flip_xz_yz(image, label):
    flip_id = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
    image = np.ascontiguousarray(image[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    label = np.ascontiguousarray(label[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    return image, label

# def intensity_shift(image):
#     shift_value = random.uniform(-0.1, 0.1)
#
#     image = image + shift_value
#     return image
#
#
# def intensity_scale(image):
#     scale_value = random.uniform(0.9, 1.1)
#
#     image = image * scale_value
#     return image

def intensity_shift(image, shift_range=0.1):
    shift = image.mean() * shift_range * (random.random() * 2 - 1)
    return image + shift

def intensity_scale(image, scale_range=(0.8, 1.2)):
    scale = random.uniform(*scale_range)
    return image * scale

def add_gaussian_noise(image, mean=0, std_dev_range=(0, 0.1)):
    std_dev = image.std() * random.uniform(*std_dev_range)
    noise = np.random.normal(mean, std_dev, image.shape)
    return np.clip(image + noise, 0, 1)

def vertical_flip(image, label):
    image = cv2.flip(image, 0)
    label = cv2.flip(label, 0)
    return image, label


def random_zoom_and_adjust_3d(image, label, zoom_range=(0.8, 1.2), seed=None):
    if seed is not None:
        random.seed(seed)

    original_shape = image.shape
    zoom_factor = random.uniform(zoom_range[0], zoom_range[1])

    new_shape = tuple(int(round(dim * zoom_factor)) for dim in original_shape)

    zoomed_image = zoom(image, zoom_factors=(zoom_factor, zoom_factor, zoom_factor), order=3)
    zoomed_label = zoom(label, zoom_factors=(zoom_factor, zoom_factor, zoom_factor), order=0)
    start = tuple((new_dim - orig_dim) // 2 for new_dim, orig_dim in zip(new_shape, original_shape))
    end = tuple(start[i] + orig_dim for i, orig_dim in enumerate(original_shape))
    adjusted_image = zoomed_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    adjusted_label = zoomed_label[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    return adjusted_image, adjusted_label
