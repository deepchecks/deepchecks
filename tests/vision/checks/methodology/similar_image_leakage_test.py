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

import torch
from hamcrest import assert_that, has_entries, close_to, equal_to

import numpy as np
from deepchecks.vision.checks import SimpleFeatureContribution, SimilarImageLeakage
from tests.checks.utils import equal_condition_result
from tests.vision.vision_conftest import *

from deepchecks.vision.utils.transformations import un_normalize_batch


def get_coco_batch_to_images_with_similar_first_images(other_batch):
    other_batch = [np.array(x) for x in other_batch[0]]

    def ret_func(batch):
        ret = [np.array(x) for x in batch[0]]

        # 5 first images will be images from another batch, but slightly different:
        # The ret[0][0][0][0] condition is so this will run only on the first batch
        if ret[0][0][0][0] == 107:
            for i in range(5):
                ret[i] = other_batch[i] + 10

        return ret

    return ret_func


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
    assert_that(result.value, equal_to(list(zip(range(64), range(64)))))


def test_similars_object_detection(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    check = SimilarImageLeakage()
    test = copy(test)
    test.batch_to_images = get_coco_batch_to_images_with_similar_first_images(next(iter(train.data_loader)))

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, equal_to(list(zip(range(5), range(5)))))


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
