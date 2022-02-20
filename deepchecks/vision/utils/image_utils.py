from typing import Tuple, Any
import PIL.Image
import cv2
import numpy as np
from torchvision.datasets import ImageFolder


def get_cv2_image(image: Any) -> np.ndarray:
    """
    Makes sure a returned image is converted into a numpy one
    :param image:
    :return:
    """
    if isinstance(image, PIL.Image.Image):
        image_np = np.array(image)
        return image_np
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise RuntimeError("Only PIL.Image and CV2 loaders currently supported!")


def cv2_loader(path: str) -> np.ndarray:
    """
    CV2 loader for PyTorch datasets
    :param path:
    :return:
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        if self.transforms is not None:
            transformed = self.transforms(image=sample, target=target)
            sample, target = transformed["image"], transformed["target"]
        else:
            if self.transform is not None:
                sample = self.transform(image=sample)['image']
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target
