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
import contextlib
import logging
import os
import typing as t
import warnings
from pathlib import Path

import numpy as np
import torch
import albumentations as A
from PIL import Image
from cv2 import cv2
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from typing_extensions import Literal

from deepchecks import vision
from deepchecks.vision.utils.detection_formatters import DetectionLabelFormatter
from deepchecks.vision.utils import ImageFormatter


__all__ = ['load_dataset', 'load_model', 'yolo_prediction_formatter', 'yolo_label_formatter', 'yolo_image_formatter']


DATA_DIR = Path(__file__).absolute().parent


def load_model(pretrained: bool = True, device: t.Union[str, torch.device] = 'cpu') -> nn.Module:
    """Load the yolov5s (version 6.1)  model and return it."""
    dev = torch.device(device) if isinstance(device, str) else device
    logger = logging.getLogger('yolov5')
    logger.disabled = True
    model = torch.hub.load('ultralytics/yolov5:v6.1', 'yolov5s',
                           pretrained=pretrained,
                           verbose=False,
                           device=dev)
    model.eval()
    logger.disabled = False
    return model


def load_dataset(
        train: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
        object_type: Literal['VisionData', 'DataLoader'] = 'DataLoader'
) -> t.Union[DataLoader, vision.VisionData]:
    """Get the COCO128 dataset and return a dataloader.

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

        A DataLoader or VisionDataset instance representing COCO128 dataset
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
            transforms=A.Compose([
                A.NoOp()
            ],
                bbox_params=A.BboxParams(format='coco')
            )
        ),
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
        return vision.VisionData(
            data_loader=dataloader,
            label_formatter=DetectionLabelFormatter(yolo_label_formatter),
            # To display images we need them as numpy array
            image_formatter=ImageFormatter(lambda batch: [np.array(x) for x in batch[0]]),
            num_classes=80,
            label_map=LABEL_MAP
        )
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')


class CocoDataset(VisionDataset):
    """An instance of PyTorch VisionData the represents the COCO128 dataset.

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

        images: t.List[Path] = sorted(self.images_dir.glob('./*.jpg'))
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
        # open image using cv2, since opening with Pillow give slightly different results based on Pillow version
        opencv_image = cv2.imread(str(self.images[idx]))
        img = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
        label_file = self.labels[idx]

        if label_file is not None:
            img_labels = [l.split() for l in label_file.open('r').read().strip().splitlines()]
            img_labels = np.array(img_labels, dtype=np.float32)
        else:
            img_labels = np.zeros((0, 5), dtype=np.float32)

        # Transform x,y,w,h in yolo format (x, y are of the image center, and coordinates are normalized) to standard
        # x,y,w,h format, where x,y are of the top left corner of the bounding box and coordinates are absolute.
        bboxes = []
        for label in img_labels:
            x, y, w, h = label[1:]
            # Note: probably the normalization loses some accuracy in the coordinates as it truncates the number,
            # leading in some cases to `y - h / 2` or `x - w / 2` to be negative
            bboxes.append(np.array([
                max((x - w / 2) * img.width, 0),
                max((y - h / 2) * img.height, 0),
                w * img.width,
                h * img.height,
                label[0]
            ]))

        img, bboxes = self.apply_transform(img, bboxes)

        # Return tensor of bboxes
        if bboxes:
            bboxes = torch.stack([torch.tensor(x) for x in bboxes])
        else:
            bboxes = torch.tensor([])
        return img, bboxes

    def apply_transform(self, img, bboxes):
        """Implement the transform in a function to be able to override it in tests."""
        if self.transforms is not None:
            # Albumentations accepts images as numpy and bboxes in defined format + class at the end
            transformed = self.transforms(image=np.array(img), bboxes=bboxes)
            img = Image.fromarray(transformed['image'])
            bboxes = transformed['bboxes']
        return img, bboxes

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

        with open(os.devnull, 'w', encoding='utf8') as f, contextlib.redirect_stdout(f):
            download_and_extract_archive(
                url,
                download_root=str(root),
                extract_root=str(root),
                md5=md5
            )

        return coco_dir, 'train2017'


def yolo_prediction_formatter(batch, model, device) -> t.List[torch.Tensor]:
    """Convert from yolo Detections object to List (per image) of Tensors of the shape [N, 6] with each row being \
    [x, y, w, h, confidence, class] for each bbox in the image."""
    return_list = []

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)

        predictions: 'ultralytics.models.common.Detections' = model.to(device)(batch[0])  # noqa: F821

        # yolo Detections objects have List[torch.Tensor] xyxy output in .pred
        for single_image_tensor in predictions.pred:
            pred_modified = torch.clone(single_image_tensor)
            pred_modified[:, 2] = pred_modified[:, 2] - pred_modified[:, 0]  # w = x_right - x_left
            pred_modified[:, 3] = pred_modified[:, 3] - pred_modified[:, 1]  # h = y_bottom - y_top
            return_list.append(pred_modified)

    return return_list


def yolo_label_formatter(batch):
    """Translate yolo label to deepchecks format."""
    # our labels return at the end, and the VisionDataset expect it at the start
    def move_class(tensor):
        return torch.index_select(tensor, 1, torch.LongTensor([4, 0, 1, 2, 3]).to(tensor.device)) \
                if len(tensor) > 0 else tensor
    return [move_class(tensor) for tensor in batch[1]]


def yolo_image_formatter(batch):
    """Convert list of PIL images to deepchecks image format."""
    # Yolo works on PIL and VisionDataset expects images as numpy arrays
    return [np.array(x) for x in batch[0]]


LABEL_MAP = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}
