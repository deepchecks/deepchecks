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
"""Module for loading a sample of the COCO dataset and the yolov5s model."""
import typing as t
from pathlib import Path

import numpy as np
import torch
# import albumentations as A
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from typing_extensions import Literal

from deepchecks import vision


__all__ = ['load_dataset']


DATA_DIR = Path(__file__).absolute().parent


def load_model(pretrained: bool = True) -> nn.Module:
    """Load the yolov5s model and return it."""
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=pretrained)
    model.eval()
    model.cpu()
    return model


def load_dataset(
    train: bool = True,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = True,
    object_type: Literal['Dataset', 'DataLoader'] = 'DataLoader'
) -> t.Union[DataLoader, vision.VisionDataset]:
    """Get the COCO dataset and return a dataloader.

    Parameters
    ----------
    train : bool
        if `True` train dataset, otherwise test dataset
    batch_size : int, default: 32
        Batch size for the dataloader.
    num_workers : int, default: 0
        Number of workers for the dataloader.
    shuffle : bool, default: False
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

        A DataLoader or VisionDataset instance representing COCO dataset
    """
    root = DATA_DIR
    coco_dir, dataset_name = CocoDataset.download_coco128(root)

    def batch_collate(batch):
        imgs, labels = zip(*batch)
        return list(imgs), list(labels)

    dataloader = DataLoader(
        dataset=CocoDataset(
            root=str(coco_dir),
            name=dataset_name,
            train=train,
            # transform=A.Compose([
            #     # TODO: what transformations we need to apply
            # ])
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=batch_collate,
        pin_memory=pin_memory,
    )

    if object_type == 'DataLoader':
        return dataloader
    elif object_type == 'Dataset':
        return vision.VisionDataset(
            data_loader=dataloader,
            label_type='object_detection'
        )
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')


class CocoDataset(VisionDataset):
    """An instance of PyTorch VisionDataset the represents the COCO dataset.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset.
    name : str
        Name of the dataset.
    train : bool
        if `True` train dataset, otherwise test dataset
    transform : Callable, optional
        A function/transforms that takes in an image and a label and returns the
        transformed versions of both.
        E.g, ``transforms.Rotate``
    target_transform : Callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : Callable, optional
        A function/transform that takes in an PIL image and returns a transformed version.
        E.g, transforms.RandomCrop
    """

    TRAIN_FRACTION = 0.5

    def __init__(
        self,
        root: str,
        name: str,
        train: bool = True,
        transform: t.Optional[t.Callable] = None,
        target_transform: t.Optional[t.Callable] = None,
        transforms: t.Optional[t.Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.train = train
        self.root = Path(root).absolute()
        self.images_dir = Path(root) / 'images' / name
        self.labels_dir = Path(root) / 'labels' / name

        images: t.List[Path] = list(self.images_dir.glob('./*.jpg'))
        labels: t.List[t.Optional[Path]] = []

        for image in images:
            label = self.labels_dir / f'{image.stem}.txt'
            labels.append(label if label.exists() else None)

        assert \
            len(images) != 0, \
            'Did not find folder with images or it was empty'
        assert \
            not all(l is None for l in labels), \
            'Did not find folder with labels or it was empty'

        train_len = int(self.TRAIN_FRACTION * len(images))

        if self.train is True:
            self.images = images[0:train_len]
            self.labels = labels[0:train_len]
        else:
            self.images = images[train_len:]
            self.labels = labels[train_len:]

    def __getitem__(self, idx: int) -> t.Tuple[Image.Image, np.ndarray]:
        """Get the image and label at the given index."""
        img = Image.open(self.images[idx]).convert('RGB')
        label_file = self.labels[idx]

        if label_file is not None:
            img_labels = [l.split() for l in label_file.open('r').read().strip().splitlines()]
            img_labels = np.array(img_labels, dtype=np.float32)
        else:
            img_labels = np.zeros((0, 5), dtype=np.float32)

        # Transform x,y,w,h in yolo format (x, y are of the image center, and coordinates are normalized) to standard
        # x,y,w,h format, where x,y are of the top left corner of the bounding box and coordinates are absolute.
        for i in range(len(img_labels)):
            x, y, w, h = img_labels[i, 1:]
            img_labels[i, 1:] = np.array([
                (x - w / 2) * img.width,
                (y - h / 2) * img.height,
                w * img.width,
                h * img.height
            ])

        if self.transforms is not None:
            img, img_labels = self.transforms(img, img_labels)

        return img, img_labels

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.images)

    @classmethod
    def download_coco128(cls, root: t.Union[str, Path]) -> t.Tuple[Path, str]:
        root = root if isinstance(root, Path) else Path(root)
        coco_dir = root / 'coco128'
        images_dir = root / 'images' / 'train2017'
        labels_dir = root / 'labels' / 'train2017'

        if not (root.exists() and root.is_dir()):
            raise RuntimeError(f'root path does not exist or is not a dir - {root}')

        if images_dir.exists() and labels_dir.exists():
            return coco_dir, 'train2017'

        url = 'https://ultralytics.com/assets/coco128.zip'
        md5 = '90faf47c90d1cfa5161c4298d890df55'

        download_and_extract_archive(
            url,
            download_root=str(root),
            extract_root=str(root),
            md5=md5
        )

        return coco_dir, 'train2017'


def yolo_wrapper(
    predictions: 'ultralytics.models.common.Detections'  # noqa: F821
) -> t.List[torch.Tensor]:
    """Convert from yolo Detections object to List (per image) of Tensors of the shape [N, 6] with each row being \
    [x, y, w, h, confidence, class] for each bbox in the image."""
    return_list = []

    # yolo Detections objects have List[torch.Tensor] xyxy output in .pred
    for single_image_tensor in predictions.pred:
        pred_modified = torch.clone(single_image_tensor)
        pred_modified[:, 2] = pred_modified[:, 2] - pred_modified[:, 0]  # w = x_right - x_left
        pred_modified[:, 3] = pred_modified[:, 3] - pred_modified[:, 1]  # h = y_bottom - y_top
        return_list.append(pred_modified)

    return return_list
