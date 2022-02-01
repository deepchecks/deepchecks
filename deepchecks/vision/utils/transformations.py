
import albumentations
import torchvision
import imgaug
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError


def get_transform_type(transformers):
    if isinstance(transformers, (albumentations.Compose, albumentations.BasicTransform)):
        return 'albumentations'
    elif isinstance(transformers, imgaug.augmenters.Augmenter):
        return 'imgaug'
    # If not albumentations and not imgaug, assuming to be pytorch (which is just plain callable without inheritance)
    else:
        return 'torch'


def get_dummy_transform(transform_type):
    if transform_type == 'albumentations':
        return albumentations.CenterCrop(1, 1)
    elif transform_type == 'imgaug':
        return imgaug.augmenters.CropToFixedSize(1, 1).augment_image
    elif transform_type == 'torch':
        return torchvision.transforms.CenterCrop(1)
    else:
        raise DeepchecksNotSupportedError(f'Not implemented for type {transform_type}')


def add_augmentation(dataset, field, aug):
    transformers = dataset.__getattribute__(field)
    transform_type = get_transform_type(transformers)

    if transform_type == 'albumentations':
        if not isinstance(aug, (albumentations.Compose, albumentations.BasicTransform)):
            raise DeepchecksValueError('')
        new_transform = albumentations.Compose([aug, transformers])
    elif transform_type == 'imgaug':
        if not isinstance(aug, imgaug.augmenters.Augmenter):
            raise DeepchecksValueError('')
        new_transform = imgaug.augmenters.Sequential([aug, transformers])
    else:
        new_transform = torchvision.transforms.Compose([aug, transformers])

    dataset.__setattribute__(field, new_transform)
