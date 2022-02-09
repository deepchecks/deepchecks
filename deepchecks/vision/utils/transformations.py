from copy import copy
from typing import Sized

import albumentations
import imgaug
import torch

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError


class ImgaugTransformations:

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
        return [
            imgaug.augmenters.MultiplyHueAndSaturation()
        ]


class AlbumentationsTransformations:

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
            # albumentations.RandomBrightnessContrast(p=1.0),
            albumentations.ShiftScaleRotate(p=1.0),
        ]
        # if data_dim == 3:
        #     augmentations.extend([
        #         albumentations.HueSaturationValue(p=1.0),
        #         albumentations.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0)
        #     ])
        return augmentations


def get_transforms_handler(transforms):
    if transforms is None:
        raise DeepchecksNotSupportedError("Underlying Dataset instance must have transform field not None")
    elif isinstance(transforms, albumentations.Compose):
        return AlbumentationsTransformations
    elif isinstance(transforms, imgaug.augmenters.Augmenter):
        return ImgaugTransformations
    else:
        raise DeepchecksNotSupportedError('Currently only imgaug and albumentations are supported')


def add_augmentation_in_start(aug, transforms):
    return get_transforms_handler(transforms).add_augmentation_in_start(aug, transforms)


def get_bbox_test_transformation(transforms):
    return get_transforms_handler(transforms).get_bbox_test_transformation()


def un_normalize_batch(tensor, mean: Sized, std: Sized, max_pixel_value: int = 255):
    dim = len(mean)
    reshape_shape = (1, 1, 1, dim)
    max_pixel_value = [max_pixel_value] * dim
    mean = torch.tensor(mean).reshape(reshape_shape)
    std = torch.tensor(std).reshape(reshape_shape)
    tensor = (tensor * std) + mean
    tensor = tensor * torch.tensor(max_pixel_value).reshape(reshape_shape)
    return tensor.numpy()
