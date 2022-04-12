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
from hashlib import md5

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from PIL import Image

from deepchecks.core import DatasetKind
from deepchecks.vision import VisionData, Context, Batch

from deepchecks.vision.datasets.detection.coco import (
    load_model as load_yolov5_model,
    load_dataset as load_coco_dataset, COCOData
)
from deepchecks.vision.datasets.classification.mnist import (
    load_model as load_mnist_net_model,
    load_dataset as load_mnist_dataset, MNISTData
)
from deepchecks.vision.vision_data import TaskType

from tests.vision.utils_tests.mnist_imgaug import mnist_dataset_imgaug
from tests.vision.assets.coco_detections_dict import coco_detections_dict
from tests.vision.assets.mnist_predictions_dict import mnist_predictions_dict


# Fix bug with torch.hub path on windows
PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent
torch.hub.set_dir(str(PROJECT_DIR))

__all__ = ['device',
           'mnist_data_loader_train',
           'mnist_dataset_train',
           'mnist_data_loader_test',
           'mnist_dataset_train_imgaug',
           'mnist_dataset_test',
           'obj_detection_images',
           'coco_train_dataloader',
           'coco_train_visiondata',
           'coco_test_dataloader',
           'coco_test_visiondata',
           'two_tuples_dataloader',
           'mnist_drifted_datasets',
           'mock_trained_yolov5_object_detection',
           'mock_trained_mnist',
           'run_update_loop',
           'mnist_train_only_images',
           'mnist_train_only_labels',
           'mnist_test_only_images',
           'mnist_train_custom_task',
           'mnist_test_custom_task',
           'coco_train_custom_task'
           ]


def _hash_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        image = Image.fromarray(image.cpu().detach().numpy().squeeze())

    image = image.resize((10, 10))
    image = image.convert('L')

    pixel_data = list(image.getdata())
    avg_pixel = sum(pixel_data) / len(pixel_data)

    bits = ''.join(['1' if (px >= avg_pixel) else '0' for px in pixel_data])
    hex_representation = str(hex(int(bits, 2)))[2:][::-1].upper()
    md_5hash = md5()
    md_5hash.update(hex_representation.encode())
    return md_5hash.hexdigest()


@pytest.fixture(scope='session')
def device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # pylint: disable=redefined-outer-name
    else:
        device = torch.device('cpu')  # pylint: disable=redefined-outer-name

    return device


@pytest.fixture(scope='session')
def mnist_data_loader_train():
    return load_mnist_dataset(train=True, object_type='DataLoader', shuffle=False)


@pytest.fixture(scope='session')
def mnist_dataset_train():
    """Return MNist dataset as VisionData object."""
    return load_mnist_dataset(train=True, object_type='VisionData', shuffle=False)


@pytest.fixture(scope='session')
def mnist_data_loader_test():
    return load_mnist_dataset(train=False, object_type='DataLoader', shuffle=False)


@pytest.fixture(scope='session')
def mnist_dataset_test():
    """Return MNist dataset as VisionData object."""
    return load_mnist_dataset(train=False, object_type='VisionData', shuffle=False)


@pytest.fixture
def mnist_drifted_datasets(mnist_dataset_train, mnist_dataset_test):  # pylint: disable=redefined-outer-name
    full_mnist = torch.utils.data.ConcatDataset([mnist_dataset_train.data_loader.dataset,
                                                 mnist_dataset_test.data_loader.dataset])
    train_dataset, test_dataset = torch.utils.data.random_split(full_mnist, [60000, 10000],
                                                                generator=torch.Generator().manual_seed(42))

    np.random.seed(42)

    def collate_test(batch):
        modified_batch = []
        for item in batch:
            image, label = item
            if label == 0:
                if np.random.randint(5) == 0:
                    modified_batch.append(item)
                else:
                    modified_batch.append((image, 1))
            else:
                modified_batch.append(item)

        return default_collate(modified_batch)

    mod_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    mod_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, collate_fn=collate_test)
    mod_train_ds = MNISTData(mod_train_loader)
    mod_test_ds = MNISTData(mod_test_loader)

    return mod_train_ds, mod_test_ds


@pytest.fixture(scope='session')
def mnist_dataset_train_imgaug():
    """Return MNist dataset as VisionData object."""
    return mnist_dataset_imgaug(train=True)


@pytest.fixture(scope='session')
def mock_trained_mnist(device):  # pylint: disable=redefined-outer-name
    class MockMnist:
        """Class of MNIST model that returns cached predictions."""

        def __init__(self, real_model):
            self.real_model = real_model

        def __call__(self, batch):
            results = []
            for img in batch:
                hash_key = _hash_image(img)
                if hash_key in mnist_predictions_dict:
                    # Predictions are saved as numpy
                    cache_pred = mnist_predictions_dict[hash_key]
                    results.append(torch.Tensor(cache_pred).to(device))
                else:
                    results.append(self.real_model(torch.stack([img]))[0])

            return torch.stack(results).to(device)

        def to(self, device):  # pylint: disable=redefined-outer-name,unused-argument
            return self

    # The MNIST model training is not deterministic, so loading a saved version of it for the tests.
    path = pathlib.Path(__file__).absolute().parent / 'models' / 'mnist.pth'
    loaded_model = load_mnist_net_model(pretrained=True, path=path).to(device)
    return MockMnist(loaded_model)


@pytest.fixture(scope='session')
def mock_trained_yolov5_object_detection(device):  # pylint: disable=redefined-outer-name

    class MockDetections:
        """Class which mocks YOLOv5 predictions object."""
        def __init__(self, dets):
            self.pred = dets

    class MockYolo:
        """Class of YOLOv5 that returns cached predictions."""
        def __init__(self, real_model):
            self.real_model = real_model

        def __call__(self, batch):
            results = []
            for img in batch:
                hash_key = _hash_image(img)
                if hash_key in coco_detections_dict:
                    results.append(coco_detections_dict[hash_key])
                else:
                    results.append(self.real_model([img]).pred[0])

            return MockDetections([x.to(device) for x in results])

        def to(self, device):  # pylint: disable=redefined-outer-name,unused-argument
            return self

    loaded_model = load_yolov5_model(device=device)
    return MockYolo(loaded_model)


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
    return load_coco_dataset(train=True, object_type='DataLoader', shuffle=False)


@pytest.fixture(scope='session')
def coco_train_visiondata():
    return load_coco_dataset(train=True, object_type='VisionData', shuffle=False)


@pytest.fixture(scope='session')
def coco_test_dataloader():
    return load_coco_dataset(train=False, object_type='DataLoader', shuffle=False)


@pytest.fixture(scope='session')
def coco_test_visiondata():
    return load_coco_dataset(train=False, object_type='VisionData', shuffle=False)


@pytest.fixture(scope='session')
def two_tuples_dataloader():
    class TwoTupleDataset(Dataset):
        def __getitem__(self, index):
            return [index, index]

        def __len__(self) -> int:
            return 8

    return DataLoader(TwoTupleDataset(), batch_size=4)


@pytest.fixture(scope='session')
def mnist_train_only_images(mnist_data_loader_train):  # pylint: disable=redefined-outer-name
    data = MNISTData(mnist_data_loader_train)
    data._label_formatter_error = 'fake error'  # pylint: disable=protected-access
    return data


@pytest.fixture(scope='session')
def mnist_train_only_labels(mnist_data_loader_train):  # pylint: disable=redefined-outer-name
    data = MNISTData(mnist_data_loader_train)
    data._image_formatter_error = 'fake error'  # pylint: disable=protected-access
    return data


@pytest.fixture(scope='session')
def mnist_test_only_images(mnist_data_loader_test):  # pylint: disable=redefined-outer-name
    data = MNISTData(mnist_data_loader_test)
    data._label_formatter_error = 'fake error'  # pylint: disable=protected-access
    return data


@pytest.fixture(scope='session')
def mnist_train_custom_task(mnist_data_loader_train):  # pylint: disable=redefined-outer-name
    class CustomTask(MNISTData):
        @property
        def task_type(self) -> TaskType:
            return TaskType.OTHER

    return CustomTask(mnist_data_loader_train, transform_field='transform')


@pytest.fixture(scope='session')
def mnist_test_custom_task(mnist_data_loader_test):  # pylint: disable=redefined-outer-name
    class CustomTask(MNISTData):
        @property
        def task_type(self) -> TaskType:
            return TaskType.OTHER

    return CustomTask(mnist_data_loader_test, transform_field='transform')


@pytest.fixture(scope='session')
def coco_train_custom_task(coco_train_dataloader):  # pylint: disable=redefined-outer-name
    class CustomTask(COCOData):
        @property
        def task_type(self) -> TaskType:
            return TaskType.OTHER

    return CustomTask(coco_train_dataloader)


def run_update_loop(dataset: VisionData):
    context: Context = Context(dataset, random_state=0)
    dataset.init_cache()
    batch_start_index = 0
    for batch in context.train:
        batch = Batch(batch, context, DatasetKind.TRAIN, batch_start_index)
        dataset.update_cache(batch)
        batch_start_index += len(batch)
