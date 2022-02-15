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
from typing import Sized, Optional

import albumentations
import imgaug.augmenters as iaa
import torch

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError

__all__ = ['get_transforms_handler', 'add_augmentation_in_start', 'un_normalize_batch',
           'ImgaugTransformations', 'AlbumentationsTransformations']


class ImgaugTransformations:
    """Class containing supporting functions for imgaug transforms."""

    @classmethod
    def add_augmentation_in_start(cls, aug, transforms):
        """Add given transformations to the start of given transforms object."""
        if not isinstance(aug, iaa.Augmenter):
            raise DeepchecksValueError(f'Transforms is of type imgaug, can\'t add to it type {type(aug).__name__}')
        return iaa.Sequential([aug, transforms])

    @classmethod
    def get_test_transformation(cls):
        """Get transformation which is affecting both image data and bbox."""
        return iaa.Rotate(rotate=(20, 30))

    @classmethod
    def get_robustness_augmentations(cls, data_dim: Optional[int] = 3):
        """Get default augmentations to use in robustness report check."""
        augmentations = [
            # Tries to be similar to output of
            # albumentations.RandomBrightnessContrast
            # Exact output is difficult
            iaa.Sequential([
                iaa.contrast.LinearContrast([0.8, 1.2]),
                iaa.color.MultiplyBrightness([0.8, 1.2])

            ], name='RandomBrightnessContrast'),
            # mimics albumentations.ShiftScaleRotate
            iaa.geometric.Affine(scale=[0.9, 1.1],
                                 translate_percent=[-0.0625, 0.0625],
                                 rotate=[-45, 45],
                                 order=1,
                                 cval=0,
                                 mode='reflect',
                                 name='ShiftScaleRotate')
        ]
        if data_dim == 3:
            augmentations.extend([
                # mimics h(p=1.0),
                iaa.WithColorspace(
                    to_colorspace='HSV',
                    from_colorspace='RGB',
                    children=[
                        # Hue
                        iaa.WithChannels(0, iaa.Add((-20, 20))),
                        # Saturation
                        iaa.WithChannels(1, iaa.Add((-30, 30))),
                        # Value
                        iaa.WithChannels(0, iaa.Add((-20, 20))),
                    ],
                    name='HueSaturationValue'
                ),
                # mimics albumentations.RGBShift
                iaa.Add(value=[-15, 15],
                        per_channel=True,
                        name='RGBShift')
            ])
        return augmentations


class AlbumentationsTransformations:
    """Class containing supporting functions for albumentations transforms."""

    @classmethod
    def add_augmentation_in_start(cls, aug, transforms):
        """Add given transformations to the start of given transforms object."""
        if not isinstance(aug, (albumentations.Compose, albumentations.BasicTransform)):
            raise DeepchecksValueError(f'Transforms is of type albumentations, can\'t add to it type '
                                       f'{type(aug).__name__}')
        # Albumentations compose contains preprocessors and another metadata needed, so we can't just create a new one,
        # so we need to copy it.
        album_compose = copy(transforms)
        album_compose.transforms = [aug, *album_compose.transforms]
        return album_compose

    @classmethod
    def get_test_transformation(cls):
        """Get transformation which is affecting both image data and bbox."""
        return albumentations.Rotate(limit=(20, 30), p=1)

    @classmethod
    def get_robustness_augmentations(cls, data_dim: Optional[int] = 3):
        """Get default augmentations to use in robustness report check."""
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
    elif isinstance(transforms, iaa.Augmenter):
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
