# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
# pylint: skip-file
import pathlib

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from deepchecks.vision.context import Context
from deepchecks.vision.datasets.classification.mnist_torch import collate_without_model as mnist_collate_without_model
from deepchecks.vision.datasets.classification.mnist_torch import deepchecks_collate as mnist_deepchecks_collate
from deepchecks.vision.datasets.classification.mnist_torch import load_dataset as load_mnist_dataset
from deepchecks.vision.datasets.classification.mnist_torch import load_model as load_mnist_model
from deepchecks.vision.datasets.detection import coco_tensorflow
from deepchecks.vision.datasets.detection.coco_torch import collate_without_model as coco_collate_without_model
from deepchecks.vision.datasets.detection.coco_torch import load_dataset as load_coco_dataset
from deepchecks.vision.datasets.segmentation.segmentation_coco import load_dataset as load_segmentation_coco_dataset
from deepchecks.vision.utils.test_utils import (replace_collate_fn_dataloader, replace_collate_fn_visiondata,
                                                un_normalize_batch)
from deepchecks.vision.vision_data import TaskType, VisionData
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper
from deepchecks.vision.vision_data.utils import set_seeds

# Fix bug with torch.hub path on windows
PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent
torch.hub.set_dir(str(PROJECT_DIR))

__all__ = ['device',
           'seed_setup',
           'mnist_iterator_visiondata_train',
           'mnist_iterator_visiondata_test',
           'mnist_dataloader_train',
           'mnist_visiondata_train',
           'mnist_dataloader_test',
           'mnist_visiondata_test',
           'obj_detection_images',
           'coco_dataloader_train',
           'coco_visiondata_train',
           'tf_coco_visiondata_train',
           'coco_dataloader_test',
           'coco_visiondata_test',
           'tf_coco_visiondata_test',
           'two_tuples_dataloader',
           'mnist_drifted_datasets',
           'run_update_loop',
           'mnist_train_only_images',
           'mnist_train_only_labels',
           'mnist_test_only_images',
           'mnist_train_custom_task',
           'mnist_test_custom_task',
           'segmentation_coco_visiondata_train',
           'segmentation_coco_visiondata_test',
           'segmentation_coco_visiondata_test_full',
           'mnist_train_very_small',
           'coco_test_only_labels',
           'coco_train_very_small',
           'mnist_train_brightness_bias',
           'coco_train_brightness_bias',
           ]


@pytest.fixture(scope='session')
def device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # pylint: disable=redefined-outer-name
    else:
        device = torch.device('cpu')  # pylint: disable=redefined-outer-name
    return device


@pytest.fixture(scope='session')
def seed_setup():
    set_seeds(42)


@pytest.fixture(scope='session')
def mnist_iterator_visiondata_train(seed_setup):
    return load_mnist_dataset(train=True, object_type='VisionData', shuffle=False, use_iterable_dataset=True,
                              n_samples=200)


@pytest.fixture(scope='session')
def mnist_iterator_visiondata_test(seed_setup):
    return load_mnist_dataset(train=False, object_type='VisionData', shuffle=False, use_iterable_dataset=True,
                              n_samples=200)


@pytest.fixture(scope='session')
def mnist_dataloader_train(seed_setup):
    return load_mnist_dataset(train=True, object_type='DataLoader', shuffle=False)


@pytest.fixture(scope='session')
def mnist_visiondata_train(seed_setup):
    """Return MNist dataset as VisionData object."""
    return load_mnist_dataset(train=True, object_type='VisionData', shuffle=False, n_samples=200)


@pytest.fixture(scope='session')
def mnist_dataloader_test(seed_setup):
    return load_mnist_dataset(train=False, object_type='DataLoader', shuffle=False)


@pytest.fixture(scope='session')
def mnist_visiondata_test(seed_setup):
    """Return MNist dataset as VisionData object."""
    return load_mnist_dataset(train=False, object_type='VisionData', shuffle=False, n_samples=200)


@pytest.fixture
def mnist_drifted_datasets(mnist_visiondata_train, mnist_visiondata_test):  # pylint: disable=redefined-outer-name
    full_mnist = torch.utils.data.ConcatDataset([mnist_visiondata_train.batch_loader.dataset,
                                                 mnist_visiondata_test.batch_loader.dataset])
    set_seeds(42)
    train_dataset, test_dataset, _ = torch.utils.data.random_split(full_mnist, [1000, 500, 68500],
                                                                   generator=torch.Generator())
    model = load_mnist_model(pretrained=True)

    def collate_drifted(data):
        raw_images = torch.stack([x[0] for x in data])
        labels = [x[1] for x in data]

        modified_labels, modified_raw_images = [], []
        for raw_image, label in zip(raw_images, labels):
            if label == 0 and np.random.randint(4) != 0:
                modified_labels.append(2)
                modified_raw_images.append(raw_image)
            elif label == 1 and np.random.randint(3) != 0:
                pass
            else:
                modified_labels.append(label)
                modified_raw_images.append(raw_image)
        modified_raw_images = torch.stack(modified_raw_images)
        predictions = model(modified_raw_images)
        predictions[:, 0] = 0
        images = modified_raw_images.permute(0, 2, 3, 1)
        images = un_normalize_batch(images, mean=(0.1307,), std=(0.3081,))
        return {'images': images, 'labels': modified_labels, 'predictions': predictions}

    mod_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                                   collate_fn=mnist_deepchecks_collate(model))
    mod_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, collate_fn=collate_drifted)
    mod_train_ds = VisionData(mod_train_loader, task_type=TaskType.CLASSIFICATION.value)
    mod_test_ds = VisionData(mod_test_loader, task_type=TaskType.CLASSIFICATION.value)
    return mod_train_ds, mod_test_ds


@pytest.fixture(scope='session')
def obj_detection_images(seed_setup):
    uris = [
        'http://images.cocodataset.org/val2017/000000397133.jpg',
        'http://images.cocodataset.org/val2017/000000037777.jpg',
        'http://images.cocodataset.org/val2017/000000252219.jpg'
    ]

    return uris


@pytest.fixture(scope='session')
def coco_dataloader_train(seed_setup):
    return load_coco_dataset(train=True, object_type='DataLoader', shuffle=False)


@pytest.fixture(scope='session')
def coco_visiondata_train(seed_setup):
    return load_coco_dataset(train=True, object_type='VisionData', shuffle=False)


@pytest.fixture(scope='session')
def tf_coco_visiondata_train(seed_setup):
    set_seeds(42)
    return coco_tensorflow.load_dataset(train=True, object_type='VisionData', shuffle=False)


@pytest.fixture(scope='session')
def coco_dataloader_test(seed_setup):
    return load_coco_dataset(train=False, object_type='DataLoader', shuffle=False)


@pytest.fixture(scope='session')
def tf_coco_visiondata_test(seed_setup):
    return coco_tensorflow.load_dataset(train=False, object_type='VisionData', shuffle=False)


@pytest.fixture(scope='session')
def coco_visiondata_test(seed_setup):
    return load_coco_dataset(train=False, object_type='VisionData', shuffle=False)


@pytest.fixture(scope='session')
def two_tuples_dataloader(seed_setup):
    class TwoTupleDataset(Dataset):
        def __getitem__(self, index):
            return [index, index]

        def __len__(self) -> int:
            return 8

    return DataLoader(TwoTupleDataset(), batch_size=4)


def _mnist_collate_only_images(data):
    return {'images': mnist_collate_without_model(data)[0]}


@pytest.fixture(scope='session')
def mnist_train_only_images(mnist_visiondata_train):  # pylint: disable=redefined-outer-name
    return replace_collate_fn_visiondata(mnist_visiondata_train, _mnist_collate_only_images)


@pytest.fixture(scope='session')
def mnist_train_only_labels(mnist_visiondata_train):  # pylint: disable=redefined-outer-name
    def collate_fn(data):
        return {'labels': mnist_collate_without_model(data)[1]}

    return replace_collate_fn_visiondata(mnist_visiondata_train, collate_fn)


@pytest.fixture(scope='session')
def mnist_test_only_images(mnist_visiondata_test):  # pylint: disable=redefined-outer-name
    return replace_collate_fn_visiondata(mnist_visiondata_test, _mnist_collate_only_images)


@pytest.fixture(scope='session')
def mnist_train_custom_task(mnist_dataloader_train):  # pylint: disable=redefined-outer-name
    loader_correct_format = replace_collate_fn_dataloader(mnist_dataloader_train, _mnist_collate_only_images)
    return VisionData(loader_correct_format, task_type=TaskType.OTHER.value, reshuffle_data=False)


@pytest.fixture(scope='session')
def mnist_test_custom_task(mnist_dataloader_test):  # pylint: disable=redefined-outer-name
    loader_correct_format = replace_collate_fn_dataloader(mnist_dataloader_test, _mnist_collate_only_images)
    return VisionData(loader_correct_format, task_type=TaskType.OTHER.value, reshuffle_data=False)


@pytest.fixture(scope='session')
def mnist_train_very_small(seed_setup):  # pylint: disable=redefined-outer-name
    return load_mnist_dataset(train=True, object_type='VisionData', shuffle=False, n_samples=5)


@pytest.fixture(scope='session')
def mnist_train_brightness_bias(mnist_visiondata_train):  # pylint: disable=redefined-outer-name
    def mnist_collate_with_bias(data):
        labels = [x[1] for x in data]
        raw_images = torch.stack([x[0] for x in data])
        tensor = raw_images.permute(0, 2, 3, 1)
        ret = un_normalize_batch(tensor, (0.1307,), (0.3081,))
        for i, label in enumerate(labels):
            ret[i] = ret[i].clip(min=5 * label, max=180 + 5 * label)
        return {'images': ret, 'labels': labels}

    return replace_collate_fn_visiondata(mnist_visiondata_train, mnist_collate_with_bias)


@pytest.fixture(scope='session')
def coco_train_brightness_bias(coco_visiondata_train):  # pylint: disable=redefined-outer-name
    def coco_collate_with_bias(data):
        raw_images = [x[0] for x in data]
        images = [np.array(x) for x in raw_images]

        def move_class(tensor):
            return torch.index_select(tensor, 1, torch.LongTensor([4, 0, 1, 2, 3]).to(tensor.device)) \
                if len(tensor) > 0 else tensor

        labels = [move_class(x[1]) for x in data]
        for i, bboxes_per_image in enumerate(labels):
            for bbox in bboxes_per_image:
                if bbox[0] > 40:
                    x, y, w, h = [round(float(n)) for n in bbox[1:]]
                    images[i][y:y + h, x:x + w] = images[i][y:y + h, x:x + w].clip(min=200)
        return {'images': images, 'labels': labels}

    return replace_collate_fn_visiondata(coco_visiondata_train, coco_collate_with_bias)


@pytest.fixture(scope='session')
def coco_train_very_small(seed_setup):  # pylint: disable=redefined-outer-name
    return load_coco_dataset(train=True, object_type='VisionData', shuffle=False, n_samples=5)


@pytest.fixture(scope='session')
def coco_test_only_labels(coco_visiondata_test):  # pylint: disable=redefined-outer-name
    def collate_fn(data):
        return {'labels': coco_collate_without_model(data)[1]}

    return replace_collate_fn_visiondata(coco_visiondata_test, collate_fn)


def run_update_loop(dataset: VisionData):
    context: Context = Context(dataset, random_state=0)
    dataset.init_cache()
    for batch in context.train:
        batch = BatchWrapper(batch, dataset.task_type, dataset.number_of_images_cached)
        dataset.update_cache(len(batch), batch.numpy_labels, batch.numpy_predictions)


@pytest.fixture(scope='session')
def segmentation_coco_visiondata_train(seed_setup):
    return load_segmentation_coco_dataset(train=True, object_type='VisionData', shuffle=False, test_mode=True)


@pytest.fixture(scope='session')
def segmentation_coco_visiondata_test(seed_setup):
    return load_segmentation_coco_dataset(train=False, object_type='VisionData', shuffle=False, test_mode=True)


@pytest.fixture(scope='session')
def segmentation_coco_visiondata_test_full(seed_setup):
    return load_segmentation_coco_dataset(train=False, object_type='VisionData', shuffle=False, test_mode=False,
                                          batch_size=10)
