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
import os
import typing as t
from pathlib import Path
from typing import Sequence

import albumentations as A
import numpy as np
import torch
import torchvision.transforms.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from typing_extensions import Literal

from deepchecks import vision
from deepchecks.vision import SegmentationData

__all__ = ['load_dataset', 'load_model', 'CocoSegmentationData', 'CocoSegmentationDataset']

DATA_DIR = Path(__file__).absolute().parent


def load_model(pretrained: bool = True, device: t.Union[str, torch.device] = 'cpu') -> nn.Module:
    """Load the lraspp_mobilenet_v3_large model and return it."""
    model = lraspp_mobilenet_v3_large(pretrained=pretrained, progress=False)
    model.eval()

    return model


class CocoSegmentationData(SegmentationData):
    """Class for loading the COCO segmentation dataset, inherits from :class:`~deepchecks.vision.SegmentationData`.

    Implement the necessary methods to load the dataset.
    """

    def batch_to_labels(self, batch):
        """Extract from the batch only the labels and return the labels in format (H, W).

        See SegmentationData for more details on format.
        """
        return batch[1]

    def infer_on_batch(self, batch, model, device):
        """Infer on a batch of images and return predictions in format (C, H, W), where C is the class_id dimension.

        See SegmentationData for more details on format.
        """
        normalized_batch = [F.normalize(img.unsqueeze(0).float()/255,
                                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) for img in batch[0]]

        predictions = [model(img)["out"].squeeze(0).detach() for img in normalized_batch]
        predictions = [torch.nn.functional.softmax(pred, dim=0) for pred in predictions]

        return predictions

    def batch_to_images(self, batch) -> Sequence[np.ndarray]:
        """Convert the batch to a list of images, where each image is a 3D numpy array in the format (H, W, C)."""
        return [tensor.numpy().transpose((1, 2, 0)) for tensor in batch[0]]


def _batch_collate(batch):
    """Get list of samples from `CocoSegmentDataset` and combine them to a batch."""
    images, masks = zip(*batch)
    return list(images), list(masks)


def load_dataset(
        train: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
        object_type: Literal['VisionData', 'DataLoader'] = 'VisionData',
        test_mode: bool = False
) -> t.Union[DataLoader, vision.VisionData]:
    """Get the COCO128 dataset and return a dataloader.

    Parameters
    ----------
    train : bool, default: True
        if `True` train dataset, otherwise test dataset
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
    test_mode: bool, default False
        whether to load this dataset in "test_mode", meaning very minimal number of images in order to use for
        unittests.

    Returns
    -------
    Union[DataLoader, VisionDataset]

        A DataLoader or VisionDataset instance representing COCO128 dataset
    """
    root = DATA_DIR
    dataset = CocoSegmentationDataset.load_or_download(root=root, train=train, test_mode=test_mode)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_batch_collate,
        pin_memory=pin_memory,
        generator=torch.Generator(),
    )

    if object_type == 'DataLoader':
        return dataloader
    elif object_type == 'VisionData':
        return CocoSegmentationData(
            data_loader=dataloader,
            num_classes=21,
            label_map=LABEL_MAP,
        )
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')


class CocoSegmentationDataset(VisionDataset):
    """An instance of PyTorch VisionData the represents the COCO128-segments dataset.

    Uses only the 21 categories used also by Pascal-VOC, in order to match the model supplied in this file,
    torchvision's deeplabv3_mobilenet_v3_large.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset.
    name : str
        Name of the dataset.
    train : bool
        if `True` train dataset, otherwise test dataset
    transforms : Callable, optional
        A function/transform that takes in an PIL image and returns a transformed version.
        E.g, transforms.RandomCrop
    """

    TRAIN_FRACTION = 0.5
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

    def __init__(
            self,
            root: str,
            name: str,
            train: bool = True,
            transforms: t.Optional[t.Callable] = None,
            test_mode: bool = False
    ) -> None:
        super().__init__(root, transforms=transforms)

        self.train = train
        self.root = Path(root).absolute()
        self.images_dir = Path(root) / 'images' / name
        self.labels_dir = Path(root) / 'labels' / name

        all_images: t.List[Path] = sorted(self.images_dir.glob('./*.jpg'))

        images: t.List[Path] = []
        labels: t.List[t.Optional[Path]] = []

        for i in range(len(all_images)):
            label = self.labels_dir / f'{all_images[i].stem}.txt'
            if label.exists():
                polygons = label.open('r').read().strip().splitlines()
                relevant_labels = [polygon.split()[0] for polygon in polygons]
                relevant_labels = [class_id for class_id in relevant_labels if int(class_id) in self.CAT_LIST]

                if len(relevant_labels) > 0:
                    images.append(all_images[i])
                    labels.append(label)

        assert len(images) != 0, 'Did not find folder with images or it was empty'
        assert not all(l is None for l in labels), 'Did not find folder with labels or it was empty'

        train_len = int(self.TRAIN_FRACTION * len(images))

        if test_mode is True:
            if self.train is True:
                self.images = images[0:5] * 2
                self.labels = labels[0:5] * 2
            else:
                self.images = images[1:6] * 2
                self.labels = labels[1:6] * 2
        else:
            if self.train is True:
                self.images = images[0:train_len]
                self.labels = labels[0:train_len]
            else:
                self.images = images[train_len:]
                self.labels = labels[train_len:]

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """Get the image and label at the given index."""
        image = Image.open(str(self.images[idx]))
        label_file = self.labels[idx]

        masks = []
        classes = []
        if label_file is not None:
            for label_str in label_file.open('r').read().strip().splitlines():
                label = np.array(label_str.split(), dtype=np.float32)
                class_id = int(label[0]) + 1
                if class_id in self.CAT_LIST:
                    # Transform normalized coordinates to un-normalized
                    coordinates = (label[1:].reshape(-1, 2) * np.array([image.width, image.height])).reshape(
                        -1).tolist()
                    # Create mask image
                    mask = Image.new('L', (image.width, image.height), 0)
                    ImageDraw.Draw(mask).polygon(coordinates, outline=1, fill=1)
                    # Add to list
                    masks.append(np.array(mask, dtype=bool))
                    classes.append(self.CAT_LIST.index(class_id))

        if self.transforms is not None:
            # Albumentations accepts images as numpy
            transformed = self.transforms(image=np.array(image), masks=masks)
            image = transformed['image']
            masks = transformed['masks']
            # Transform masks to tensor of (num_masks, H, W)
            if masks:
                if isinstance(masks[0], np.ndarray):
                    masks = [torch.from_numpy(m) for m in masks]
                masks = torch.stack(masks)
            else:
                masks = torch.empty((0, 3))

        # Fake grayscale to rgb because model can't process grayscale:
        if image.shape[0] == 1:
            image = torch.stack([image[0], image[0], image[0]])

        ret_label = np.zeros((image.shape[1], image.shape[2]))
        ret_label_mask = np.zeros(ret_label.shape)
        for i in range(len(classes)):
            mask = np.logical_and(np.logical_not(ret_label_mask), np.array(masks[i]))
            ret_label_mask = np.logical_or(ret_label_mask, mask)
            ret_label += classes[i] * mask

        return image, torch.as_tensor(ret_label)

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.images)

    @classmethod
    def load_or_download(cls, root: Path, train: bool, test_mode: bool) -> 'CocoSegmentationDataset':
        """Load or download the coco128 dataset with segment annotations."""
        extract_dir = root / 'coco128segments'
        coco_dir = root / 'coco128segments' / 'coco128'
        folder = 'train2017'

        if not coco_dir.exists():
            url = 'https://ultralytics.com/assets/coco128-segments.zip'
            md5 = 'e29ec06014d1e06b58b6ffe651c0b34f'

            with open(os.devnull, 'w', encoding='utf8') as f, contextlib.redirect_stdout(f):
                download_and_extract_archive(
                    url,
                    download_root=str(root),
                    extract_root=str(extract_dir),
                    md5=md5
                )

            try:
                # remove coco128 README.txt so that it does not come in docs
                os.remove("coco128segments/coco128/README.txt")
            except:  # pylint: disable=bare-except # noqa
                pass
        return CocoSegmentationDataset(coco_dir, folder, train=train, transforms=A.Compose([ToTensorV2()]),
                                       test_mode=test_mode)


_ORIG_LABEL_MAP = {
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
    12: 'street sign',
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
    26: 'hat',
    27: 'backpack',
    28: 'umbrella',
    29: 'shoe',
    30: 'eye glasses',
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
    45: 'plate',
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
    66: 'mirror',
    67: 'dining table',
    68: 'window',
    69: 'desk',
    70: 'toilet',
    71: 'door',
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
    83: 'blender',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
    91: 'hair brush',
}

LABEL_MAP = {CocoSegmentationDataset.CAT_LIST.index(k): v for k, v in _ORIG_LABEL_MAP.items() if k in
             CocoSegmentationDataset.CAT_LIST}
LABEL_MAP[0] = 'background'
