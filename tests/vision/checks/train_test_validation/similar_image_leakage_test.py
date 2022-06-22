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

import numpy as np
from hamcrest import assert_that, equal_to, greater_than, has_length
from PIL import Image
from torch.utils.data import DataLoader

from deepchecks.vision.checks import SimilarImageLeakage
from deepchecks.vision.utils.test_utils import get_modified_dataloader
from tests.base.utils import equal_condition_result


def test_no_similar_object_detection(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    check = SimilarImageLeakage()

    # Act
    result = check.run(train, test, device=device)

    # Assert
    assert_that(result.value, equal_to([]))


def test_no_similar_classification(mnist_dataset_train, mnist_dataset_test, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_test
    check = SimilarImageLeakage(hash_size=32, similarity_threshold=0.02)

    # Act
    result = check.run(train, test, n_samples=500, random_state=42, device=device)

    # Assert
    assert_that(result.value, equal_to([]))


def test_all_identical_object_detection(coco_train_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    check = SimilarImageLeakage()

    # Act
    result = check.run(train, test, device=device)

    # Assert
    assert_that(set(result.value), equal_to(set(list(zip(range(64), range(64))))))
    assert_that(result.display, has_length(greater_than(0)))


def test_all_identical_object_detection_without_display(coco_train_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    check = SimilarImageLeakage()

    # Act
    result = check.run(train, test, device=device, with_display=False)

    # Assert
    assert_that(set(result.value), equal_to(set(list(zip(range(64), range(64))))))
    assert_that(result.display, has_length(0))


def test_similar_object_detection(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    check = SimilarImageLeakage()
    test = copy(test)

    def get_modification_func():
        other_dataset = train.data_loader.dataset

        def mod_func(orig_dataset, idx):
            if idx in range(5):
                data, label = other_dataset[idx]
                return Image.fromarray(np.clip(np.array(data, dtype=np.uint16) + 50, 0, 255).astype(np.uint8)), label
            if idx == 30:  # Also test something that is not in the same order
                data, label = other_dataset[0]
                return Image.fromarray(np.clip(np.array(data, dtype=np.uint16) + 50, 0, 255).astype(np.uint8)), label
            else:
                return orig_dataset[idx]

        return mod_func

    test._data_loader = get_modified_dataloader(test, get_modification_func())

    # Act
    result = check.run(train, test, device=device)

    # Assert
    assert_that(set(result.value), equal_to(set(zip(range(5), range(5))).union({(0, 30)})))


def test_train_test_condition_pass(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    condition_value = 5
    check = SimilarImageLeakage().add_condition_similar_images_less_or_equal(condition_value)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test, device=device)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Found 0 similar images between train and test datasets',
        name=f'Number of similar images between train and test is less or equal to {condition_value}'
    ))


def test_train_test_condition_fail(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    condition_value = 5
    check = SimilarImageLeakage().add_condition_similar_images_less_or_equal(condition_value)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test, device=device)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Number of similar images between train and test is less or equal to {condition_value}',
        details='Number of similar images between train and test datasets: 64'
    ))
