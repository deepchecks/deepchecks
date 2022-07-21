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
import numpy as np
import PIL.Image as pilimage
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from typing_extensions import Literal

from deepchecks import vision

__all__ = ['classification_dataset_from_directory', 'SimpleClassificationDataset', 'SimpleClassificationData']


def classification_dataset_from_directory(
        root: str,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
        object_type: Literal['VisionData', 'DataLoader'] = 'DataLoader',
        **kwargs
) -> t.Union[t.Tuple[t.Union[DataLoader, vision.ClassificationData]], t.Union[DataLoader, vision.ClassificationData]]:
    """Load a simple classification dataset.

    The function expects that the data within the root folder
    to be structured one of the following ways:

        - root/
            - class1/
                image1.jpeg

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
    t.Union[t.Tuple[t.Union[DataLoader, vision.ClassificationData]], t.Union[DataLoader, vision.ClassificationData]]
        A DataLoader or VisionDataset instance or tuple representing a single dataset or train and test datasets.
    """

    def batch_collate(batch):
        imgs, labels = zip(*batch)
        return list(imgs), list(labels)

    root_path = Path(root).absolute()
    if not (root_path.exists() and root_path.is_dir()):
        raise ValueError(f'{root_path} - path does not exist or is not a folder')

    roots_of_datasets = []
    if root_path.joinpath('train').exists():
        roots_of_datasets.append(root_path.joinpath('train'))
    if root_path.joinpath('test').exists():
        roots_of_datasets.append(root_path.joinpath('test'))
    if len(roots_of_datasets) == 0:
        roots_of_datasets.append(root_path)

    result = []
    for dataset_root in roots_of_datasets:
        dataset = SimpleClassificationDataset(root=str(dataset_root), **kwargs)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=batch_collate,
            pin_memory=pin_memory,
            generator=torch.Generator()
        )
        if object_type == 'DataLoader':
            result.append(dataloader)
        elif object_type == 'VisionData':
            result.append(SimpleClassificationData(data_loader=dataloader, label_map=dataset.reverse_classes_map))
        else:
            raise TypeError(f'Unknown value of object_type - {object_type}')
    return tuple(result) if len(result) > 1 else result[0]


class SimpleClassificationDataset(VisionDataset):
    """Simple VisionDataset type for the classification tasks.

    The current class expects that data within the root folder
    will be structured the following way:

        - root/
            - class1/
                image1.jpeg

    Otherwise, exception will be raised.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset.
    transforms : Callable, optional
        A function/transform that takes in an PIL image and returns a transformed version.
        E.g, transforms.RandomCrop
    transform : Callable, optional
        A function/transforms that takes in an image and a label and returns the
        transformed versions of both.
        E.g, ``transforms.Rotate``
    target_transform : Callable, optional
        A function/transform that takes in the target and transforms it.
    image_extension : str, default 'jpg'
        images format
    """

    def __init__(
            self,
            root: str,
            transforms: t.Optional[t.Callable] = None,
            transform: t.Optional[t.Callable] = None,
            target_transform: t.Optional[t.Callable] = None,
            image_extension: str = 'jpg'
    ) -> None:
        self.root_path = Path(root).absolute()

        if not (self.root_path.exists() and self.root_path.is_dir()):
            raise ValueError(f'{self.root_path} - path does not exist or is not a folder')

        super().__init__(str(self.root_path), transforms, transform, target_transform)
        self.image_extension = image_extension.lower()
        self.images = sorted(self.root_path.glob(f'*/*.{self.image_extension}'))

        if len(self.images) == 0:
            raise ValueError(f'{self.root_path} - is empty or has incorrect structure')

        classes = {img.parent.name for img in self.images}
        classes = sorted(list(classes))

        # class label -> class index
        self.classes_map = t.cast(t.Dict[str, int], dict(zip(classes, range(len(classes)))))
        # class index -> class  label
        self.reverse_classes_map = t.cast(t.Dict[int, str], {
            v: k
            for k, v in self.classes_map.items()
        })

    def __getitem__(self, index: int) -> t.Tuple[np.ndarray, int]:
        """Get the image and label at the given index."""
        image_file = self.images[index]
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = self.classes_map[image_file.parent.name]

        if self.transforms is not None:
            transformed = self.transforms(image=image, target=target)
            image, target = transformed['image'], transformed['target']
        else:
            if self.transform is not None:
                image = self.transform(image=image)['image']
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.images)


class SimpleClassificationData(vision.ClassificationData):
    """Simple ClassificationData type, matches the data returned by SimpleClassificationDataset getitem."""

    def batch_to_images(
            self,
            batch: t.Tuple[t.Sequence[np.ndarray], t.Sequence[int]]
    ) -> t.Sequence[np.ndarray]:
        """Extract the images from a batch of data."""
        images, _ = batch
        return images

    def batch_to_labels(
            self,
            batch: t.Tuple[t.Sequence[pilimage.Image], t.Sequence[int]]
    ) -> torch.Tensor:
        """Extract the labels from a batch of data."""
        _, labels = batch
        return torch.Tensor(labels).long()

    def get_classes(self, batch_labels: t.Union[t.List[torch.Tensor], torch.Tensor]):
        """Get a labels batch and return classes inside it."""
        return batch_labels.reshape(-1, 1).tolist()
