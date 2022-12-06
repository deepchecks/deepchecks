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
from collections import OrderedDict
from hashlib import md5

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from deepchecks.vision import BatchWrapper, Context, VisionData
from deepchecks.vision.datasets.classification.mnist import load_dataset as load_mnist_dataset
from deepchecks.vision.datasets.classification.mnist import load_model as load_mnist_net_model
from deepchecks.vision.datasets.detection.coco import load_dataset as load_coco_dataset
from deepchecks.vision.datasets.detection.coco import load_model as load_yolov5_model
from deepchecks.vision.datasets.segmentation.segmentation_coco import load_dataset as load_segmentation_coco_dataset
from deepchecks.vision.datasets.segmentation.segmentation_coco import load_model as load_segmentation_coco_model
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.utils.test_utils import replace_collate_fn_function
from deepchecks.vision.utils.transformations import un_normalize_batch
from tests.vision.assets.coco_detections_dict import coco_detections_dict
from tests.vision.assets.mnist_predictions_dict import mnist_predictions_dict
from tests.vision.utils_tests.mnist_imgaug import mnist_dataset_imgaug

# Fix bug with torch.hub path on windows
PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent
torch.hub.set_dir(str(PROJECT_DIR))

__all__ = ['device',
           'mnist_dataloader_train',
           'mnist_visiondata_train',
           'mnist_dataloader_test',
           'mnist_dataset_train_imgaug',
           'mnist_visiondata_test',
           'obj_detection_images',
           'coco_dataloader_train',
           'coco_visiondata_train',
           'coco_dataloader_test',
           'coco_visiondata_test',
           'two_tuples_dataloader',
           'mnist_drifted_datasets',
           'mock_trained_yolov5_object_detection',
           'mock_mnist_model',
           'run_update_loop',
           'mnist_train_only_images',
           'mnist_train_only_labels',
           'mnist_test_only_images',
           'mnist_train_custom_task',
           'mnist_test_custom_task',
           'coco_train_custom_task',
           'segmentation_coco_visiondata_train',
           'segmentation_coco_visiondata_test',
           'segmentation_coco_visiondata_test_full',
           'trained_segmentation_deeplabv3_mobilenet_model'
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
def mnist_dataloader_train():
    return load_mnist_dataset(train=True, object_type='DataLoader', shuffle=False)


@pytest.fixture(scope='session')
def mnist_visiondata_train():
    """Return MNist dataset as VisionData object."""
    return load_mnist_dataset(train=True, object_type='VisionData', shuffle=False)


@pytest.fixture(scope='session')
def mnist_dataloader_test():
    return load_mnist_dataset(train=False, object_type='DataLoader', shuffle=False)


@pytest.fixture(scope='session')
def mnist_visiondata_test():
    """Return MNist dataset as VisionData object."""
    return load_mnist_dataset(train=False, object_type='VisionData', shuffle=False)


@pytest.fixture
def mnist_drifted_datasets(mnist_visiondata_train, mnist_visiondata_test):  # pylint: disable=redefined-outer-name
    full_mnist = torch.utils.data.ConcatDataset([mnist_visiondata_train.data_loader.dataset,
                                                 mnist_visiondata_test.data_loader.dataset])
    train_dataset, test_dataset = torch.utils.data.random_split(full_mnist, [60000, 10000],
                                                                generator=torch.Generator().manual_seed(42))
    np.random.seed(42)

    def collate_test(data):
        raw_images = torch.stack([x[0] for x in data])
        labels = [x[1] for x in data]
        images = raw_images.permute(0, 2, 3, 1)
        images = un_normalize_batch(images, mean=(0.1307,), std=(0.3081,))

        modified_batch = []
        for item in zip(images, labels):
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
    mod_train_ds = VisionData(mod_train_loader, task_type=TaskType.CLASSIFICATION.value)
    mod_test_ds = VisionData(mod_test_loader, task_type=TaskType.CLASSIFICATION.value)
    return mod_train_ds, mod_test_ds


@pytest.fixture(scope='session')
def mnist_dataset_train_imgaug():
    """Return MNist dataset as VisionData object."""
    return mnist_dataset_imgaug(train=True)


@pytest.fixture(scope='session')
def mock_mnist_model(device):  # pylint: disable=redefined-outer-name
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
def coco_dataloader_train():
    return load_coco_dataset(train=True, object_type='DataLoader', shuffle=False)


@pytest.fixture(scope='session')
def coco_visiondata_train():
    return load_coco_dataset(train=True, object_type='VisionData', shuffle=False)


@pytest.fixture(scope='session')
def coco_dataloader_test():
    return load_coco_dataset(train=False, object_type='DataLoader', shuffle=False)


@pytest.fixture(scope='session')
def coco_visiondata_test():
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
def mnist_train_only_images(mnist_dataloader_train):  # pylint: disable=redefined-outer-name
    def collate_fn(data):
        raw_images = torch.stack([x[0] for x in data])
        images = raw_images.permute(0, 2, 3, 1)
        images = un_normalize_batch(images, mean=(0.1307,), std=(0.3081,))
        return {'images': images}

    new_loader = replace_collate_fn_function(mnist_dataloader_train, collate_fn)
    return VisionData(dynamic_loader=new_loader, task_type=TaskType.CLASSIFICATION.value)


@pytest.fixture(scope='session')
def mnist_train_only_labels(mnist_dataloader_train):  # pylint: disable=redefined-outer-name
    def collate_fn(data):
        labels = [x[1] for x in data]
        return {'labels': labels}

    new_loader = replace_collate_fn_function(mnist_dataloader_train, collate_fn)
    return VisionData(dynamic_loader=new_loader, task_type=TaskType.CLASSIFICATION.value)


@pytest.fixture(scope='session')
def mnist_test_only_images(mnist_dataloader_test):  # pylint: disable=redefined-outer-name
    def collate_fn(data):
        raw_images = torch.stack([x[0] for x in data])
        images = raw_images.permute(0, 2, 3, 1)
        images = un_normalize_batch(images, mean=(0.1307,), std=(0.3081,))
        return {'images': images}

    new_loader = replace_collate_fn_function(mnist_dataloader_test, collate_fn)
    return VisionData(dynamic_loader=new_loader, task_type=TaskType.CLASSIFICATION.value)


@pytest.fixture(scope='session')
def mnist_train_custom_task(mnist_dataloader_train):  # pylint: disable=redefined-outer-name
    return VisionData(mnist_dataloader_train, task_type=TaskType.OTHER.value)


@pytest.fixture(scope='session')
def mnist_test_custom_task(mnist_dataloader_test):  # pylint: disable=redefined-outer-name
    return VisionData(mnist_dataloader_test, task_type=TaskType.OTHER.value)


@pytest.fixture(scope='session')
def coco_train_custom_task(coco_dataloader_train):  # pylint: disable=redefined-outer-name
    return VisionData(coco_train_custom_task, task_type=TaskType.OTHER.value)


def run_update_loop(dataset: VisionData):
    context: Context = Context(dataset, random_state=0)
    dataset.init_cache()
    for i, batch in enumerate(context.train):
        batch = BatchWrapper(batch, dataset)
        dataset.update_cache(len(batch), batch.numpy_labels, batch.numpy_predictions)


@pytest.fixture(scope='session')
def segmentation_coco_visiondata_train():
    return load_segmentation_coco_dataset(train=True, object_type='VisionData', shuffle=False, test_mode=True)


@pytest.fixture(scope='session')
def segmentation_coco_visiondata_test():
    return load_segmentation_coco_dataset(train=False, object_type='VisionData', shuffle=False, test_mode=True)


@pytest.fixture(scope='session')
def segmentation_coco_visiondata_test_full():
    return load_segmentation_coco_dataset(train=False, object_type='VisionData', shuffle=False, test_mode=False,
                                          batch_size=10)


@pytest.fixture(scope='session')
def trained_segmentation_deeplabv3_mobilenet_model():
    class MockDeepLab:
        """Class of DeepLabV3MobileNet model that returns cached predictions."""

        def __init__(self, real_model):
            self.real_model = real_model
            self._cache = {}

        def __call__(self, batch):
            results = []
            for img in batch:
                img_to_hash = ((img + img.min()) / img.max() * 255).type(torch.uint8)
                img_to_hash = torch.transpose(img_to_hash, 0, 2)
                hash_key = _hash_image(img_to_hash)
                if self._cache.get(hash_key) is not None:
                    results.append(self._cache[hash_key])
                else:
                    # results.append(self.real_model(torch.stack([img]))[0])
                    res = self.real_model(img.unsqueeze(0))['out'].squeeze(0)
                    results.append(res)
                    self._cache[hash_key] = res

            return OrderedDict([('out', torch.stack(results))])

        def to(self, device):  # pylint: disable=redefined-outer-name,unused-argument
            return self

    model = load_segmentation_coco_model()
    return MockDeepLab(model)
