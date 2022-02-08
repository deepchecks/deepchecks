from copy import copy
from typing import Sized

import albumentations
import imgaug
import torch

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError


def add_augmentation_in_start(aug, transforms):
    """Return new object which contains given augmentation in the start and then the rest of transforms."""
    transform_type = get_transform_type(transforms)
    if transform_type == 'albumentations':
        if not isinstance(aug, (albumentations.Compose, albumentations.BasicTransform)):
            raise DeepchecksValueError(f'Transforms is of type albumentations, can\'t add to it type {type(aug)}')
        # Albumentations compose contains preprocessors and another metadata needed, so we can't just create a new one,
        # so we need to copy it.
        album_compose = copy(transforms)
        album_compose.transforms = [aug, *album_compose.transforms]
        return album_compose
    elif transform_type == 'imgaug':
        if not isinstance(aug, imgaug.augmenters.Augmenter):
            raise DeepchecksValueError(f'Transforms is of type imgaug, can\'t add to it type {type(aug)}')
        return imgaug.augmenters.Sequential([aug, transforms])
    else:
        raise DeepchecksNotSupportedError(f'Not implemented for type {transform_type}')


def get_transform_type(transforms):
    if transforms is None:
        raise DeepchecksNotSupportedError("Underlying Dataset instance must have transform field not None")
    elif isinstance(transforms, albumentations.Compose):
        return 'albumentations'
    elif isinstance(transforms, imgaug.augmenters.Augmenter):
        return 'imgaug'
    else:
        raise DeepchecksNotSupportedError('Currently only imgaug and albumentations are supported')


def un_normalize_batch(tensor, mean: Sized, std: Sized, max_pixel_value: int = 255):
    dim = len(mean)
    reshape_shape = (1, 1, 1, dim)
    max_pixel_value = [max_pixel_value] * dim
    mean = torch.tensor(mean).reshape(reshape_shape)
    std = torch.tensor(std).reshape(reshape_shape)
    tensor = (tensor * std) + mean
    tensor = tensor * torch.tensor(max_pixel_value).reshape(reshape_shape)
    return tensor.numpy()
