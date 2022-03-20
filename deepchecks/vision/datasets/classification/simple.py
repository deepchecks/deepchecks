# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains implementation of the simple classification dataset."""
import typing as t
from pathlib import Path

import cv2
import PIL.Image as pilimage
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from typing_extensions import Literal

from deepchecks import vision


__all__ = ['load_dataset', 'SimpleClassificationDataset']


def load_dataset(
    root: str,
    train: bool = True,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
    object_type: Literal['VisionData', 'DataLoader'] = 'DataLoader',
    **kwargs
) -> t.Union[DataLoader, vision.ClassificationData]:
    """Load a simple classification dataset.

    The function expects that data within root folder
    will be structured in a next way:

        - root/
            - train/
                - class1/
                    image1.jpeg
            - test/
                - class1/
                    image1.jpeg

    Parameters
    ----------
    root : str
        path to the data
    train : bool
        if `True` load the train dataset, otherwise load the test dataset
    batch_size : int, default: 32
        Batch size for the dataloader.
    num_workers : int, default: 0
        Number of workers for the dataloader.
    shuffle : bool, default: True
        Whether to shuffle the dataset.
    pin_memory : bool, default: True
        If ``True``, the data loader will copy Tensors
        into CUDA pinned memory before returning them.
    object_type : Literal['Dataset', 'DataLoader'], default: 'DataLoader'
        type of the return value. If 'Dataset', :obj:`deepchecks.vision.VisionDataset`
        will be returned, otherwise :obj:`torch.utils.data.DataLoader`

    Returns
    -------
    Union[DataLoader, VisionDataset]

        A DataLoader or VisionDataset instance representing the dataset
    """

    def batch_collate(batch):
        imgs, labels = zip(*batch)
        return list(imgs), list(labels)

    dataloader = DataLoader(
        dataset=SimpleClassificationDataset(root=root, train=train, **kwargs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=batch_collate,
        pin_memory=pin_memory,
        generator=torch.Generator()
    )

    if object_type == 'DataLoader':
        return dataloader
    elif object_type == 'VisionData':
        return SimpleClassificationData(data_loader=dataloader)
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')


class SimpleClassificationDataset(VisionDataset):
    """Simple VisionDataset type for the classification tasks.

    Current class expects that data within root folder
    will be structured in a next way:

        - root/
            - train/
                - class1/
                    image1.jpeg
            - test/
                - class1/
                    image1.jpeg

    Otherwise exception will be raised.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset.
    train : bool
        if `True` train dataset, otherwise test dataset
    transforms : Callable, optional
        A function/transform that takes in an PIL image and returns a transformed version.
        E.g, transforms.RandomCrop
    transform : Callable, optional
        A function/transforms that takes in an image and a label and returns the
        transformed versions of both.
        E.g, ``transforms.Rotate``
    target_transform : Callable, optional
        A function/transform that takes in the target and transforms it.
    image_extension : str, default 'JPEG'
        images format
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transforms: t.Optional[t.Callable] = None,
        transform: t.Optional[t.Callable] = None,
        target_transform: t.Optional[t.Callable] = None,
        image_extension: str = 'JPEG'
    ) -> None:
        self.root_path = Path(root).absolute()

        if not (self.root_path.exists() and self.root_path.is_dir()):
            raise ValueError(f'{self.root_path} - path does not exist or is not a folder')

        super().__init__(str(self.root_path), transforms, transform, target_transform)
        self.image_extension = image_extension.lower()

        if train is True:
            self.images = sorted(self.root_path.glob(f'train/*/*.{self.image_extension}'))
        else:
            self.images = sorted(self.root_path.glob(f'test/*/*.{self.image_extension}'))

        if len(self.images) == 0:
            raise ValueError(f'{self.root_path} - is empty or has incorrect structure')

        # class label -> class index
        self.classes_map = t.cast(t.Dict[str, int], {
            img.parent.name: index
            for index, img in enumerate(self.images)
        })
        # class index -> class label
        self.reverse_classes_map = t.cast(t.Dict[int, str],{
            v: k
            for k, v in self.classes_map.items()
        })

    def __getitem__(self, index: int) -> t.Tuple[pilimage.Image, int]:
        """Get the image and label at the given index."""
        image_file = self.images[index]
        image = cv2.imread(str(image_file))
        image = pilimage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image, self.classes_map[image_file.parent.name]

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.images)


class SimpleClassificationData(vision.ClassificationData):
    """Simple ClassificationData type."""

    def batch_to_images(
        self,
        batch: t.Tuple[t.Sequence[pilimage.Image], t.Sequence[int]]
    ) -> t.Sequence[np.ndarray]:
        """Extract the images from a batch of data."""
        images, _ = batch
        return [np.array(img) for img in images]

    def batch_to_labels(
        self,
        batch: t.Tuple[t.Sequence[pilimage.Image], t.Sequence[int]]
    ) -> torch.Tensor:
        """Extract the labels from a batch of data."""
        _, labels = batch
        return torch.Tensor([l for l in labels])

    def infer_on_batch(self, batch, model, device) -> torch.Tensor:
        """Return the predictions of the model on a batch of data."""
        labels = self.batch_to_labels(batch)
        return model.to(device)(labels.to(device))
