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
from torch.utils.data.dataloader import default_collate

from deepchecks.vision import VisionData
from deepchecks.vision.datasets.detection.coco import (
    load_model as load_yolov5_model,
    load_dataset as load_coco_dataset
)
from deepchecks.vision.datasets.classification.mnist import (
    load_model as load_mnist_net_model,
    load_dataset as load_mnist_dataset
)
from deepchecks.vision.utils import ClassificationLabelFormatter

from tests.vision.utils_tests.mnist_imgaug import mnist_dataset_imgaug

# Fix bug with torch.hub path on windows
PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent
torch.hub.set_dir(str(PROJECT_DIR))

__all__ = ['device',
           'mnist_data_loader_train',
           'mnist_dataset_train',
           'mnist_data_loader_test',
           'mnist_dataset_train_imgaug',
           'mnist_dataset_test',
           'trained_mnist',
           'trained_yolov5_object_detection',
           'obj_detection_images',
           'coco_train_dataloader',
           'coco_train_visiondata',
           'coco_test_dataloader',
           'coco_test_visiondata',
           'two_tuples_dataloader',
           'mnist_drifted_datasets'
           ]


@pytest.fixture(scope='session')
def device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # pylint: disable=redefined-outer-name
    else:
        device = torch.device('cpu')  # pylint: disable=redefined-outer-name

    return device


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


@pytest.fixture
def mnist_drifted_datasets(mnist_dataset_train, mnist_dataset_test):  # pylint: disable=redefined-outer-name
    full_mnist = torch.utils.data.ConcatDataset([mnist_dataset_train.get_data_loader().dataset,
                                                 mnist_dataset_test.get_data_loader().dataset])
    train_dataset, test_dataset = torch.utils.data.random_split(full_mnist, [60000, 10000],
                                                                generator=torch.Generator().manual_seed(42))

    np.random.seed(42)

    def collate_test(batch):
        modified_batch = []
        for item in batch:
            _, label = item
            if (label == 0) and (np.random.randint(10) == 0):
                modified_batch.append(item)
            else:
                modified_batch.append(item)

        return default_collate(modified_batch)

    mod_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    mod_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, collate_fn=collate_test)
    mod_train_ds = VisionData(mod_train_loader, label_formatter=ClassificationLabelFormatter())
    mod_test_ds = VisionData(mod_test_loader, label_formatter=ClassificationLabelFormatter())

    return mod_train_ds, mod_test_ds


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
def trained_yolov5_object_detection(device):  # pylint: disable=redefined-outer-name
    return load_yolov5_model(device=device)


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
def two_tuples_dataloader():
    class TwoTupleDataset(Dataset):
        def __getitem__(self, index):
            return [index, index]

        def __len__(self) -> int:
            return 8

    return DataLoader(TwoTupleDataset(), batch_size=4)
