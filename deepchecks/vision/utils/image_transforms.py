import albumentations as A
import numpy as np
from albumentations.augmentations import functional as F
from albumentations.pytorch import ToTensorV2


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
        img_ = F.normalize(img, mean=[0., 0., 0.,], std=self.std,
                           max_pixel_value=1.0)
        img_ = F.normalize(img_, mean=self.mean, std=[1., 1., 1.,],
                           max_pixel_value=1.0)
        img_ = (img_ * 255).astype(np.int32)
        return img_


class ReverseToTensorV2(A.BasicTransform):
    def __init__(self, totensor_transform: A.pytorch.ToTensorV2):
        super(ReverseToTensorV2, self).__init__(always_apply=totensor_transform.always_apply,
                                                p=totensor_transform.p)
        self.transpose_mask = totensor_transform.transpose_mask

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if len(img.shape) != 3:
            raise ValueError("Inverse ToTensorV2 only supports images in CHW format")

        return img.transpose(1, 2, 0)

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(1, 2, 0)
        return mask