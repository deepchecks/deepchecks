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
import pathlib
import numpy as np

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from deepchecks.vision.dataset import VisionData
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision.datasets.detection.coco import (
    load_model as load_yolov5_model,
    load_dataset as load_coco_dataset
)
from deepchecks.vision.datasets.classification.mnist import (
    load_model as load_mnist_net_model,
    load_dataset as load_mnist_dataset
)
from deepchecks.vision.utils.detection_formatters import DetectionLabelFormatter
from deepchecks.vision.utils.image_formatters import ImageFormatter
from tests.vision.utils_tests.mnist_imgaug import mnist_dataset_imgaug
from tests.vision.assets.coco_detections_dict import coco_detections_dict

# Fix bug with torch.hub path on windows
PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent
torch.hub.set_dir(str(PROJECT_DIR))


__all__ = ['simple_formatter',
           'mnist_data_loader_train',
           'mnist_dataset_train',
           'mnist_data_loader_test',
           'mnist_dataset_train_imgaug',
           'mnist_dataset_test',
           'trained_mnist',
           'trained_yolov5_object_detection',
           'mock_trained_yolov5_object_detection',
           'obj_detection_images',
           'coco_train_dataloader',
           'coco_train_visiondata',
           'coco_test_dataloader',
           'coco_test_visiondata',
           'two_tuples_dataloader',
        ]


def _batch_collate(batch):
    imgs, labels, idx = zip(*batch)
    return list(imgs), list(labels), list(idx)

@pytest.fixture(scope='session')
def simple_formatter():
    def formatter(batch, model, _):
        return model(batch[2])
    return formatter

@pytest.fixture(scope='session')
def mnist_data_loader_train():
    return load_mnist_dataset(train=True, object_type='DataLoader')


@pytest.fixture(scope='session')
def mnist_dataset_train():
    """Return MNist dataset as VisionData object."""
    return load_mnist_dataset(train=True, object_type='VisionData')


@pytest.fixture(scope='session')
def mnist_data_loader_test():
    return load_mnist_dataset(train=False, object_type='DataLoader')


@pytest.fixture(scope='session')
def mnist_dataset_test():
    """Return MNist dataset as VisionData object."""
    return load_mnist_dataset(train=False, object_type='VisionData')


@pytest.fixture(scope='session')
def trained_mnist():
    # The MNIST model training is not deterministic, so loading a saved version of it for the tests.
    path = pathlib.Path(__file__).absolute().parent / 'models' / 'mnist.pth'
    return load_mnist_net_model(pretrained=True, path=path)


@pytest.fixture(scope='session')
def mnist_dataset_train_imgaug():
    """Return MNist dataset as VisionData object."""
    return mnist_dataset_imgaug(train=True)


@pytest.fixture(scope='session')
def trained_yolov5_object_detection():
    return load_yolov5_model()

@pytest.fixture(scope='session')
def mock_trained_yolov5_object_detection():
    class FakeYolo():
        def __call__(self, batch):
            detections = []
            for im_id in batch:
                detections.append(coco_detections_dict[im_id])
            return detections
    return FakeYolo()

@pytest.fixture(scope='session')
def obj_detection_images():
    uris = [
        'http://images.cocodataset.org/val2017/000000397133.jpg',
        'http://images.cocodataset.org/val2017/000000037777.jpg',
        'http://images.cocodataset.org/val2017/000000252219.jpg'
    ]

    return uris

@pytest.fixture(scope='session')
def coco_train_dataloader():
    return load_coco_dataset(train=True, object_type='DataLoader')


@pytest.fixture(scope='session')
def coco_train_visiondata():
    return load_coco_dataset(train=True, object_type='VisionData')

@pytest.fixture(scope='session')
def coco_train_visiondata():
    train_dataset = load_coco_dataset(train=True, object_type='DataLoader').dataset
    class TrainDataset(Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, idx):
            return (train_dataset[idx][0], train_dataset[idx][1], f'train_{idx}')
    test_dataloader = DataLoader(TrainDataset(), shuffle=True, batch_size=32, collate_fn=_batch_collate,   
                                      generator=torch.Generator())
    test_visiondata = VisionData(test_dataloader,
                                 image_formatter=ImageFormatter(lambda batch: [np.array(x) for x in batch[0]]),
                                 label_formatter=DetectionLabelFormatter(coco.yolo_label_formatter),
                                 num_classes=80, label_map=coco.LABEL_MAP)
    return test_visiondata

@pytest.fixture(scope='session')
def coco_test_dataloader():
    return load_coco_dataset(train=False, object_type='DataLoader')

@pytest.fixture(scope='session')
def coco_test_visiondata():
    test_dataset = load_coco_dataset(train=False, object_type='DataLoader').dataset
    class TestDataset(Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, idx):
            return (test_dataset[idx][0], test_dataset[idx][1], f'test_{idx}')
    test_dataloader = DataLoader(TestDataset(), shuffle=True, batch_size=32, collate_fn=_batch_collate,   
                                      generator=torch.Generator())
    test_visiondata = VisionData(test_dataloader,
                                 image_formatter=ImageFormatter(lambda batch: [np.array(x) for x in batch[0]]),
                                 label_formatter=DetectionLabelFormatter(coco.yolo_label_formatter),
                                 num_classes=80, label_map=coco.LABEL_MAP)
    return test_visiondata

@pytest.fixture(scope='session')
def two_tuples_dataloader():
    class TwoTupleDataset(Dataset):
        def __getitem__(self, index):
            return [index, index]

        def __len__(self) -> int:
            return 8

    return DataLoader(TwoTupleDataset(), batch_size=4)
