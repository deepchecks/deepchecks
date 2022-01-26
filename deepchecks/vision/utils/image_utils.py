from typing import Tuple, Any

import PIL.Image
import cv2
import numpy as np
import albumentations as A
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import StandardTransform


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
