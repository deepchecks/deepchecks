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

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from deepchecks.vision.datasets.detection.coco import (
    load_model as load_yolov5_model,
    load_dataset as load_coco_dataset
)
from deepchecks.vision.datasets.classification.mnist import (
    load_model as load_mnist_net_model,
    load_dataset as load_mnist_dataset
)
from tests.vision.utils_tests.mnist_imgaug import mnist_dataset_imgaug

# Fix bug with torch.hub path on windows
PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent
torch.hub.set_dir(str(PROJECT_DIR))


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
    return load_mnist_net_model()


@pytest.fixture(scope='session')
def mnist_dataset_train_imgaug():
    """Return MNist dataset as VisionData object."""
    return mnist_dataset_imgaug(train=True)


@pytest.fixture(scope='session')
def trained_yolov5_object_detection():
    return load_yolov5_model()


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
def coco_test_dataloader():
    return load_coco_dataset(train=False, object_type='DataLoader')


@pytest.fixture(scope='session')
def coco_test_visiondata():
    return load_coco_dataset(train=False, object_type='VisionData')


@pytest.fixture(scope='session')
def three_tuples_dataloader():
    class ThreeTupleDataset(Dataset):
        def __getitem__(self, index):
            return [index, index, index]

        def __len__(self) -> int:
            return 8

    return DataLoader(ThreeTupleDataset(), batch_size=4)


@pytest.fixture(scope='session')
def two_tuples_dataloader():
    class TwoTupleDataset(Dataset):
        def __getitem__(self, index):
            return [index, index]

        def __len__(self) -> int:
            return 8

    return DataLoader(TwoTupleDataset(), batch_size=4)
