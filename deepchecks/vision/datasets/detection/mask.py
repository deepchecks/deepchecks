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
"""Module for loading the mask detection dataset and its pre-calculated predictions.

    The mask dataset is a dataset of various images with people wearing masks, people not wearing masks and people
    wearing masks incorrectly. The dataset is used for object detection, and was downloaded from
    https://www.kaggle.com/datasets/andrewmvd/face-mask-detection, licenced under CC0.
"""
import contextlib
import hashlib
import json
import os
import typing as t
import urllib.request
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from bs4 import BeautifulSoup
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import transforms
from typing_extensions import Literal

from deepchecks import vision
from deepchecks.vision import DetectionData
from deepchecks.vision.vision_data import IndicesSequentialSampler

__all__ = ['load_dataset', 'load_model', 'MaskData', 'MaskDataset']


DATA_DIR = Path(__file__).absolute().parent


class MaskPrecalculatedModel(nn.Module):
    """Model that returns pre-calculated predictions for the mask detection dataset."""

    def __init__(self, device: t.Union[str, torch.device] = 'cpu'):
        super().__init__()
        self._pred_dict_url = 'https://ndownloader.figshare.com/files/38116641'
        pred_dict_path = os.path.join(DATA_DIR, 'pred_dict.json')
        urllib.request.urlretrieve(self._pred_dict_url, pred_dict_path)
        with open(pred_dict_path, 'r', encoding='utf8') as f:
            self._pred_dict = json.load(f)

        self._device = device

    def forward(self, images: t.Sequence[torch.Tensor]) -> t.Sequence[torch.Tensor]:
        image_hashes = [self._hash_image((img.cpu().numpy() if isinstance(img, torch.Tensor) else img))
                        for img in images]
        return [torch.tensor(self._pred_dict[image_hash]).to(self._device) for image_hash in image_hashes]

    @staticmethod
    def _hash_image(img):
        return hashlib.sha1(img).hexdigest()


def load_model(device: t.Union[str, torch.device] = 'cpu') -> nn.Module:
    """Load the pre-calculated prediction model and return it."""
    dev = torch.device(device) if isinstance(device, str) else device

    return MaskPrecalculatedModel(device=dev)


class MaskData(DetectionData):
    """Class for loading the mask dataset, inherits from :class:`~deepchecks.vision.DetectionData`.

    Implement the necessary methods to load the dataset.
    """

    def batch_to_labels(self, batch) -> List[torch.Tensor]:
        """Convert the batch to a list of labels."""

        def extract_dict(in_dict):
            return torch.concat([in_dict['labels'].reshape((-1, 1)), in_dict['boxes']], axis=1)

        return [extract_dict(tensor) for tensor in batch[1]]

    def batch_to_images(self, batch) -> Iterable[np.ndarray]:
        """Convert the batch to a list of images."""
        return [np.array(x.permute(1, 2, 0)) * 255 for x in batch[0]]

    def infer_on_batch(self, batch, model, device: t.Union[str, torch.device] = 'cpu') -> List[torch.Tensor]:
        """Infer on a batch using the given model."""
        return model(batch[0])


def _batch_collate(batch):
    return tuple(zip(*batch))


def load_dataset(
        time_index: int = 0,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
        object_type: Literal['VisionData', 'DataLoader'] = 'DataLoader'
) -> t.Union[DataLoader, vision.VisionData]:
    """Get the mask dataset and return a dataloader.

    Parameters
    ----------
    time_index : int, default: 0
        Select the moment in time to load the dataset from. 0 is the training set, and each subsequent number is a
        different time in the production dataset. Last time step is 59.
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

        A DataLoader or VisionDataset instance representing mask dataset
    """
    root = DATA_DIR
    mask_dir = MaskDataset.download_mask(root)
    time_to_sample_dict = MaskDataset.get_time_to_sample_dict(root)
    time = list(time_to_sample_dict.keys())[time_index]
    samples_to_use = time_to_sample_dict[time]

    if shuffle:
        sampler = torch.utils.data.SubsetRandomSampler(samples_to_use, generator=torch.Generator())
    else:
        sampler = IndicesSequentialSampler(samples_to_use)

    dataloader = DataLoader(
        dataset=MaskDataset(
            mask_dir=str(mask_dir),
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        ),
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_batch_collate,
        pin_memory=pin_memory,
    )

    if object_type == 'DataLoader':
        return dataloader
    elif object_type == 'VisionData':
        return MaskData(
            data_loader=dataloader,
            num_classes=3,
            label_map=LABEL_MAP,
            dataset_name=f'Mask Dataset at time {time}',
        )
    else:
        raise TypeError(f'Unknown value of object_type - {object_type}')


class MaskDataset(VisionDataset):
    """Dataset for the mask dataset. Loads the images and labels from the dataset."""

    def __init__(self, mask_dir, *args, **kwargs):
        """Initialize the dataset."""
        super().__init__(mask_dir, *args, **kwargs)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, 'images'))))

    def __getitem__(self, idx):
        """Get the image and labels at the given index."""
        # load images ad masks
        file_image = 'maksssksksss' + str(idx) + '.png'
        file_label = 'maksssksksss' + str(idx) + '.xml'
        img_path = os.path.join(os.path.join(self.root, 'images'), file_image)
        label_path = os.path.join(os.path.join(self.root, 'annotations'), file_label)
        img = Image.open(img_path).convert('RGB')
        # Generate Label
        target = self._generate_target(idx, label_path)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.imgs)

    @staticmethod
    def _generate_box(obj):

        xmin = int(obj.find('xmin').text)
        ymin = int(obj.find('ymin').text)
        xmax = int(obj.find('xmax').text)
        ymax = int(obj.find('ymax').text)

        return [xmin, ymin, xmax - xmin, ymax - ymin]

    @staticmethod
    def _generate_label(obj):
        if obj.find('name').text == 'with_mask':
            return 1
        elif obj.find('name').text == 'mask_weared_incorrect':
            return 2
        return 0

    @staticmethod
    def _generate_target(image_id, file):
        with open(file, encoding='utf8') as f:
            data = f.read()
            soup = BeautifulSoup(data, 'xml')
            objects = soup.find_all('object')

            # Bounding boxes for objects
            # In coco format, bbox = [xmin, ymin, width, height]
            # In pytorch, the input should be [xmin, ymin, xmax, ymax]
            boxes = []
            labels = []
            for i in objects:
                boxes.append(MaskDataset._generate_box(i))
                labels.append(MaskDataset._generate_label(i))
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Labels (In my case, I only one class: target class or background)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            # img_id to Tensor
            img_id = torch.tensor([image_id])
            # Annotation is in dictionary format
            target = {'boxes': boxes, 'labels': labels, 'image_id': img_id}

            return target

    @classmethod
    def download_mask(cls, root: t.Union[str, Path]) -> Path:
        """Download mask and returns the root path and folder name."""
        root = root if isinstance(root, Path) else Path(root)
        mask_dir = Path(os.path.join(root, 'mask'))
        img_path = Path(os.path.join(mask_dir, 'images'))
        label_path = Path(os.path.join(mask_dir, 'annotations'))

        if not (root.exists() and root.is_dir()):
            raise RuntimeError(f'root path does not exist or is not a dir - {root}')

        if img_path.exists() and label_path.exists():
            return mask_dir

        url = 'https://figshare.com/ndownloader/files/38115927'
        md5 = '64b8f1d3036f3445557a8619f0400f6e'

        with open(os.devnull, 'w', encoding='utf8') as f, contextlib.redirect_stdout(f):
            download_and_extract_archive(
                url,
                download_root=str(mask_dir),
                extract_root=str(mask_dir),
                md5=md5,
                filename='mask.zip'
            )

        return mask_dir

    @classmethod
    def get_time_to_sample_dict(cls, root: t.Union[str, Path]) -> t.Dict[int, t.List[int]]:
        """Return a dictionary of time to sample."""
        time_dict_url = 'https://figshare.com/ndownloader/files/38116608'

        root = root if isinstance(root, Path) else Path(root)
        time_to_sample_dict_path = Path(os.path.join(root, 'time_to_sample_dict.json'))
        if not time_to_sample_dict_path.exists():
            urllib.request.urlretrieve(time_dict_url, time_to_sample_dict_path)

        with open(time_to_sample_dict_path, 'r', encoding='utf8') as f:
            return json.load(f)


LABEL_MAP = {2: 'Improperly Worn Mask',
             1: 'Properly Worn Mask',
             0: 'No Mask'}
