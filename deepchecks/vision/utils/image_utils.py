from typing import Tuple, Any

import PIL.Image
import cv2
import numpy as np
import albumentations as A
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder


def get_cv2_image(image: Any) -> np.ndarray:
    if isinstance(image, PIL.Image.Image):
        image_np = np.array(image)
        return image_np
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise RuntimeError("Only PIL.Image and CV2 loaders currently supported!")

def cv2_loader(path: str) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class CheckLoader:
    def __init__(self, transform: A.BasicTransform):
        self.transform = transform

    def __call__(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # no apply augmentation
        img = self.transform(img)
        return img

class AlbumentationImageFolder(ImageFolder):
    def __init__(self, *args, **kwargs):
        """
        Overrides initialization method to replace default loader with OpenCV loader
        :param args:
        :param kwargs:
        """
        super(AlbumentationImageFolder, self).__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        overrides __getitem__ to be compatible to albumentations
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = get_cv2_image(sample)
        if self.transform is not None:
            sample = self.transform(image=sample)['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

# TODO move augmentations to own file
from albumentations import functional as F

class UnNormalize(A.ImageOnlyTransform):
    def __init__(self, normalize_transform: A.Normalize):
        super(UnNormalize, self).__init__(always_apply=normalize_transform.always_apply,
                                               p=1.0)
        self.mean = [-m for m in normalize_transform.mean]
        self.std = [1/s for s in normalize_transform.std]
        self.max_pixel_value = normalize_transform.max_pixel_value

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        :param img:
        :param params:
        :return:
        """
        img_ = F.normalize(img, [0., 0., 0.,], self.std, self.max_pixel_value)
        img_ = F.normalize(img_, self.mean, [1., 1., 1.,], self.max_pixel_value)
        # img_ *= 255
        return img_

class ReverseToTensorV2(A.BasicTransform):
    def __init__(self, totensor_transform: A.pytorch.ToTensorV2):
        super(ReverseToTensorV2, self).__init__(always_apply=totensor_transform.always_apply,
                                                p=totensor_transform.p)
        self.transpose_mask = totensor_transform.transpose_mask

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img: torch.Tensor, **params) -> np.ndarray:
        if len(img.shape) != 3:
            raise ValueError("Inverse ToTensorV2 only supports images in CHW format")

        return img.transpose(1, 2, 0).cpu().detach().numpy()

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(1, 2, 0)
        return mask.cpu().detach().numpy()


class AlbumentationsTransformWrapper:
    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, img, *args, **kwargs):
        return self.transform(image=np.array(img))['image']

    def insert(self, transform: A.BasicTransform):
        self.transform = A.Compose([transform] + self.transform.transforms[1:])

class AlbumentationsTransformsWrapper:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, image, *args, **kwargs):
        image = np.array(image)
        transformed = self.transforms(image=image) #, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        return transformed_image

    def insert(self, transform: A.BasicTransform):
        self.transforms = A.Compose([transform] + self.transforms.transforms[1:])
