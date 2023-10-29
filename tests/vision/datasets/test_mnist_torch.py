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
import time

from hamcrest import assert_that, calling, equal_to, instance_of, is_not, raises
from torch.utils.data import DataLoader

from deepchecks.vision import VisionData
from deepchecks.vision.datasets.classification.mnist_torch import (MNIST_DIR, MODEL_PATH, MnistModel, load_dataset,
                                                                   load_model)


def test_dataset_load():
    dataloader = load_dataset(object_type="DataLoader", n_samples=100)
    assert_that(dataloader, instance_of(DataLoader))
    assert_that(MNIST_DIR.exists() and MNIST_DIR.is_dir())
    assert_that(dataloader.dataset._check_exists() is True)


def test_deepchecks_dataset_load():
    dataloader = load_dataset(object_type='DataLoader', n_samples=100)
    dataset = load_dataset(object_type='VisionData', n_samples=100)
    assert_that(dataset, instance_of(VisionData))
    assert_that(dataloader, instance_of(DataLoader))


def test__load_dataset__func_with_unknow_object_type_parameter():
    assert_that(calling(load_dataset).with_args(object_type="<unknonw>"), raises(TypeError))


def test_pretrained_model_load():
    if MODEL_PATH.exists():
        model = load_model().real_model
        assert_that(model.training is False)
        assert_that(model, instance_of(MnistModel))
    else:
        model = load_model().real_model
        assert_that(model.training is False)
        assert_that(model, instance_of(MnistModel))
        assert_that(MODEL_PATH.exists() and MODEL_PATH.is_file())
        # to verify loading from the file
        test_pretrained_model_load()


def test_iterable_dataloader():
    loader = load_dataset(object_type='DataLoader', use_iterable_dataset=True, n_samples=100, batch_size=50)
    batch = next(iter(loader))
    assert_that(batch[0].shape, equal_to((50, 1, 28, 28)))
    assert_that(calling(len).with_args(loader),
                raises(TypeError, r'object of type \'IterableTorchMnistDataset\' has no len()'))


def test_iterable_visiondata():
    vision_data = load_dataset(object_type='VisionData', use_iterable_dataset=True, n_samples=100, batch_size=50)
    batch = next(iter(vision_data))
    assert_that(batch['images'].shape, equal_to((50, 28, 28, 1)))


def test_iterable_visiondata_with_shuffle():
    assert_that(calling(load_dataset).with_args(object_type='DataLoader', use_iterable_dataset=True, shuffle=True,
                                                n_samples=100),
                raises(ValueError,
                       r'DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle=True'))


def test_regular_visiondata_with_shuffle():
    vision_data = load_dataset(object_type='VisionData', use_iterable_dataset=False, n_samples=100, shuffle=False)
    batch = next(iter(vision_data))
    vision_data_again = load_dataset(object_type='VisionData', use_iterable_dataset=False, n_samples=100, shuffle=False)
    batch_again = next(iter(vision_data_again))
    vision_data_shuffled = load_dataset(object_type='VisionData', use_iterable_dataset=False, n_samples=100,
                                        shuffle=True)
    batch_shuffled = next(iter(vision_data_shuffled))
    vision_data_shuffled_again = load_dataset(object_type='VisionData', use_iterable_dataset=False, n_samples=100,
                                              shuffle=True)
    batch_shuffled_again = next(iter(vision_data_shuffled_again))

    assert_that(batch['labels'], is_not(equal_to(batch_shuffled['labels'])))
    assert_that(batch['labels'], equal_to(batch_again['labels']))
    assert_that(batch_shuffled_again['labels'], is_not(equal_to(batch_shuffled['labels'])))
