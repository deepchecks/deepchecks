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
import abc
import typing as t
from copy import copy

import albumentations
import imgaug.augmenters as iaa
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.vision.vision_data import TaskType

__all__ = ['get_transforms_handler', 'un_normalize_batch', 'AbstractTransformations',
           'ImgaugTransformations', 'AlbumentationsTransformations']


class AbstractTransformations(abc.ABC):
    """Abstract class for supporting functions for various transforms."""

    is_transforming_labels = True

    @classmethod
    @abc.abstractmethod
    def add_augmentation_in_start(cls, aug, transforms):
        """Add given transformations to the start of given transforms object."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def get_test_transformation(cls):
        """Get transformation which is affecting both image data and bbox."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def get_robustness_augmentations(cls, data_dim: t.Optional[int] = 3) -> t.List[t.Any]:
        """Get default augmentations to use in robustness report check."""
        raise NotImplementedError()


class ImgaugTransformations(AbstractTransformations):
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
    def get_robustness_augmentations(cls, data_dim: t.Optional[int] = 3) -> t.List[iaa.Augmenter]:
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


class AlbumentationsTransformations(AbstractTransformations):
    """Class containing supporting functions for albumentations transforms."""

    @classmethod
    def add_augmentation_in_start(cls, aug, transforms: albumentations.Compose):
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
    def get_robustness_augmentations(cls, data_dim: t.Optional[int] = 3) -> t.List[albumentations.BasicTransform]:
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


class TorchTransformations(AbstractTransformations):
    """Class containing supporting functions for torch transforms."""

    @classmethod
    def add_augmentation_in_start(cls, aug, transforms: T.Compose):
        """Add given transformations to the start of given transforms object."""
        if isinstance(aug, (albumentations.Compose, albumentations.BasicTransform)):
            alb_aug = aug

            class TorchWrapper:
                def __call__(self, image):
                    if isinstance(image, Image.Image):
                        return T.ToPILImage()(alb_aug(image=np.array(image))['image'])
                    elif isinstance(image, torch.Tensor):
                        image = image.cpu().detach().numpy()
                        return T.ToTensor()(alb_aug(image=image)['image'])
                    return alb_aug(image=image)['image']
            aug = TorchWrapper()
        elif not isinstance(aug, torch.nn.Module):
            raise DeepchecksValueError(f'Transforms is of type torch, can\'t add to it type '
                                       f'{type(aug).__name__}')
        return T.Compose([aug, *transforms.transforms])

    @classmethod
    @abc.abstractmethod
    def get_test_transformation(cls):
        """Get transformation which is affecting image data."""
        return AlbumentationsTransformations.get_test_transformation()

    @classmethod
    @abc.abstractmethod
    def get_robustness_augmentations(cls, data_dim: t.Optional[int] = 3) -> t.List[albumentations.BasicTransform]:
        """Get default augmentations to use in robustness report check."""
        return AlbumentationsTransformations.get_robustness_augmentations(data_dim)


class TorchTransformationsBbox(TorchTransformations):
    """Class containing supporting functions for torch transforms (not including image shifting)."""

    is_transforming_labels = False

    @classmethod
    def get_robustness_augmentations(cls, data_dim: t.Optional[int] = 3) -> t.List[albumentations.BasicTransform]:
        """Get default augmentations to use in robustness report check (without image shift)."""
        augs = super().get_robustness_augmentations(data_dim=data_dim)
        return filter(lambda aug: not isinstance(aug, albumentations.DualTransform), augs)


def get_transforms_handler(transforms, task_type: TaskType) -> t.Type[AbstractTransformations]:
    """Return the appropriate transforms handler based on type of given transforms."""
    if transforms is None:
        raise DeepchecksNotSupportedError('Underlying Dataset instance must have transform field not None')
    elif isinstance(transforms, albumentations.Compose):
        return AlbumentationsTransformations
    elif isinstance(transforms, T.Compose):
        if task_type == TaskType.OBJECT_DETECTION:
            return TorchTransformationsBbox
        return TorchTransformations
    elif isinstance(transforms, iaa.Augmenter):
        return ImgaugTransformations
    else:
        raise DeepchecksNotSupportedError('Currently only imgaug, albumentations and torch are supported')


def un_normalize_batch(tensor: torch.Tensor, mean: t.Sized, std: t.Sized, max_pixel_value: int = 255):
    """Apply un-normalization on a tensor in order to display an image."""
    dim = len(mean)
    reshape_shape = (1, 1, 1, dim)
    max_pixel_value = [max_pixel_value] * dim
    mean = torch.tensor(mean, device=tensor.device).reshape(reshape_shape)
    std = torch.tensor(std, device=tensor.device).reshape(reshape_shape)
    tensor = (tensor * std) + mean
    tensor = tensor * torch.tensor(max_pixel_value, device=tensor.device).reshape(reshape_shape)
    return tensor.cpu().detach().numpy()
