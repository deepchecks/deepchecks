import random

import albumentations
import imgaug
import numpy
import torch

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError


class TransformWrapper:

    def __init__(self, transforms):
        self.transform_type = get_transform_type(transforms)
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        return self.transforms(*args, **kwargs)

    def get_dummy_transform(self):
        if self.transform_type == 'albumentations':
            return albumentations.CenterCrop(1, 1)
        elif self.transform_type == 'imgaug':
            return imgaug.augmenters.CropToFixedSize(1, 1).augment_image
        else:
            raise DeepchecksNotSupportedError(f'Not implemented for type {self.transform_type}')

    def add_augmentation_in_start(self, aug):
        if self.transform_type == 'albumentations':
            if not isinstance(aug, (albumentations.Compose, albumentations.BasicTransform)):
                raise DeepchecksValueError(f'Transforms is of type albumentations, can\'t add to it type {type(aug)}')
            self.transforms = albumentations.Compose([aug, self.transforms])
        elif self.transform_type == 'imgaug':
            if not isinstance(aug, imgaug.augmenters.Augmenter):
                raise DeepchecksValueError(f'Transforms is of type imgaug, can\'t add to it type {type(aug)}')
            self.transforms = imgaug.augmenters.Sequential([aug, self.transforms])
        else:
            raise DeepchecksNotSupportedError(f'Not implemented for type {self.transform_type}')


def get_transform_type(transforms):
    if transforms is None:
        raise DeepchecksNotSupportedError("Underlying Dataset instance must have transform field not None")
    elif isinstance(transforms, (albumentations.Compose, albumentations.BasicTransform)):
        return 'albumentations'
    elif isinstance(transforms, imgaug.augmenters.Augmenter):
        return 'imgaug'
    else:
        raise DeepchecksNotSupportedError('Currently only imgaug and albumentations are supported')

