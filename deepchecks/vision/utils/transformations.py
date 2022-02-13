# ----------------------------------------------------------------------------
# Copyright (C) 2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module for defining functions related to vision transforms."""
from copy import copy
from typing import Sized

import albumentations
import imgaug
import torch

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError


__all__ = ['get_transforms_handler', 'add_augmentation_in_start', 'un_normalize_batch']


class ImgaugTransformations:
    """Class containing supporting functions for imgaug transforms."""

    @classmethod
    def add_augmentation_in_start(cls, aug, transforms):
        if not isinstance(aug, imgaug.augmenters.Augmenter):
            raise DeepchecksValueError(f'Transforms is of type imgaug, can\'t add to it type {type(aug)}')
        return imgaug.augmenters.Sequential([aug, transforms])

    @classmethod
    def get_test_transformation(cls):
        return imgaug.augmenters.Rotate(rotate=(20, 30))

    @classmethod
    def get_robustness_augmentations(cls, data_dim):
        augmentations = [
            imgaug.augmenters.MultiplyHueAndSaturation()
        ]
        if data_dim == 3:
            # TODO add RGB augmentations
            pass
        return augmentations


class AlbumentationsTransformations:
    """Class containing supporting functions for albumentations transforms."""

    @classmethod
    def add_augmentation_in_start(cls, aug, transforms):
        if not isinstance(aug, (albumentations.Compose, albumentations.BasicTransform)):
            raise DeepchecksValueError(f'Transforms is of type albumentations, can\'t add to it type {type(aug)}')
        # Albumentations compose contains preprocessors and another metadata needed, so we can't just create a new one,
        # so we need to copy it.
        album_compose = copy(transforms)
        album_compose.transforms = [aug, *album_compose.transforms]
        return album_compose

    @classmethod
    def get_test_transformation(cls):
        return albumentations.Rotate(limit=(20, 30), p=1)

    @classmethod
    def get_robustness_augmentations(cls, data_dim):
        augmentations = [
            albumentations.RandomBrightnessContrast(p=1.0),
            albumentations.ShiftScaleRotate(p=1.0),
        ]
        if data_dim == 3:
            augmentations.extend([
                albumentations.HueSaturationValue(p=1.0),
                albumentations.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0)
            ])
        return augmentations


def get_transforms_handler(transforms):
    """Return the appropriate transforms handler based on type of given transforms."""
    if transforms is None:
        raise DeepchecksNotSupportedError('Underlying Dataset instance must have transform field not None')
    elif isinstance(transforms, albumentations.Compose):
        return AlbumentationsTransformations
    elif isinstance(transforms, imgaug.augmenters.Augmenter):
        return ImgaugTransformations
    else:
        raise DeepchecksNotSupportedError('Currently only imgaug and albumentations are supported')


def add_augmentation_in_start(aug, transforms):
    """Add given augmentation to the first place in the transforms."""
    return get_transforms_handler(transforms).add_augmentation_in_start(aug, transforms)


def un_normalize_batch(tensor, mean: Sized, std: Sized, max_pixel_value: int = 255):
    """Apply un-normalization on a tensor in order to display an image."""
    dim = len(mean)
    reshape_shape = (1, 1, 1, dim)
    max_pixel_value = [max_pixel_value] * dim
    mean = torch.tensor(mean).reshape(reshape_shape)
    std = torch.tensor(std).reshape(reshape_shape)
    tensor = (tensor * std) + mean
    tensor = tensor * torch.tensor(max_pixel_value).reshape(reshape_shape)
    return tensor.cpu().numpy()
