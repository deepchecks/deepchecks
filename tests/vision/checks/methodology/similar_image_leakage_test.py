# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Test functions of the VISION train test label drift."""
from copy import copy

from hamcrest import assert_that, equal_to

import numpy as np
from deepchecks.vision.checks import SimilarImageLeakage
from tests.checks.utils import equal_condition_result

from torch.utils.data import DataLoader
from PIL import Image


def mock_dataloader(vision_data, other_dataset, shuffle=False):
    """Create a mock dataloader that replaces several images with brighter images from other_dataset."""

    class MockDataset:
        """A Mock dataset object that replaces several images with brighter images from other_dataset."""

        def __init__(self, orig_dataset):
            self._orig_dataset = orig_dataset

        def __getitem__(self, idx):
            if idx in range(5):
                data, label = other_dataset[idx]
                return Image.fromarray(np.clip(np.array(data, dtype=np.uint16) + 50, 0, 255).astype(np.uint8)), label
            else:
                data, label = self._orig_dataset[idx]
            return data, label

        def __len__(self):
            return len(self._orig_dataset)

    # Create a copy of the original dataloader, using the new dataset
    props = vision_data._get_data_loader_props(vision_data.data_loader)
    props['dataset'] = MockDataset(vision_data.data_loader.dataset)
    props['shuffle'] = shuffle
    data_loader = DataLoader(**props)
    data_loader, _ = vision_data._get_data_loader_sequential(data_loader)
    return data_loader


def test_no_similars_object_detection(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    check = SimilarImageLeakage()

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, equal_to([]))


def test_all_identical_object_detection(coco_train_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    check = SimilarImageLeakage()

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(set(result.value), equal_to(set(list(zip(range(64), range(64))))))


def test_similar_object_detection(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    check = SimilarImageLeakage()
    test = copy(test)
    test._data_loader = mock_dataloader(test, train.data_loader.dataset)

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(set(result.value), equal_to(set(zip(range(5), range(5)))))


def test_train_test_condition_pass(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    condition_value = 5
    check = SimilarImageLeakage().add_condition_similar_images_not_more_than(condition_value)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        name=f'Number of similar images between train and test is not greater than {condition_value}'
    ))


def test_train_test_condition_fail(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    condition_value = 5
    check = SimilarImageLeakage().add_condition_similar_images_not_more_than(condition_value)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Number of similar images between train and test is not greater than {condition_value}',
        details='Number of similar images between train and test datasets: 64'
    ))
