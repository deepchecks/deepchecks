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
import pandas as pd
from hamcrest import assert_that, close_to, equal_to, has_entries

from deepchecks.vision.checks import SimpleFeatureContribution
from deepchecks.vision.utils.transformations import un_normalize_batch
from tests.checks.utils import equal_condition_result
from tests.vision.vision_conftest import *


def mnist_batch_to_images_with_bias(batch):
    """Create function which inverse the data normalization."""
    tensor = batch[0]
    tensor = tensor.permute(0, 2, 3, 1)
    ret = un_normalize_batch(tensor, (0.1307,), (0.3081,))
    for i, label in enumerate(batch[1]):
        ret[i] = ret[i].clip(min=5 * label, max=180 + 5 * label)
    return ret


def get_coco_batch_to_images_with_bias(label_formatter):
    def ret_func(batch):
        ret = [np.array(x) for x in batch[0]]
        for i, labels in enumerate(label_formatter(batch)):
            for label in labels:
                if label[0] > 40:
                    x, y, w, h = [round(float(n)) for n in label[1:]]
                    ret[i][y:y + h, x:x + w] = ret[i][y:y + h, x:x + w].clip(min=200)
        return ret

    return ret_func


def get_coco_batch_to_images_with_bias_one_class(label_formatter):
    def ret_func(batch):
        ret = [np.array(x) for x in batch[0]]
        for i, labels in enumerate(label_formatter(batch)):
            for label in labels:
                if label[0] == 74:
                    x, y, w, h = [round(float(n)) for n in label[1:]]
                    ret[i][y:y + h, x:x + w] = ret[i][y:y + h, x:x + w].clip(min=200)
        return ret

    return ret_func


def test_is_float_column():
    col = pd.Series([1, 2, 3, 4, 5])
    assert_that(SimpleFeatureContribution.is_float_column(col), equal_to(False))

    col = pd.Series(['a', 'b', 'c'])
    assert_that(SimpleFeatureContribution.is_float_column(col), equal_to(False))

    col = pd.Series(['a', 'b', 5.5])
    assert_that(SimpleFeatureContribution.is_float_column(col), equal_to(False))

    col = pd.Series([1, 2, 3, 4, 5], dtype='float')
    assert_that(SimpleFeatureContribution.is_float_column(col), equal_to(False))

    col = pd.Series([1, 2, 3, 4, 5.5], dtype='float64')
    assert_that(SimpleFeatureContribution.is_float_column(col), equal_to(True))


def test_no_drift_classification(mnist_dataset_train):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = SimpleFeatureContribution(per_class=False, random_state=42)

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.08, 0.005)}),
        'test': has_entries({'Brightness': close_to(0.08, 0.005)}),
        'train-test difference': has_entries({'Brightness': equal_to(0)})
    }))


def test_drift_classification(mnist_dataset_train, mnist_dataset_test):
    mnist_dataset_test.batch_to_images = mnist_batch_to_images_with_bias

    train, test = mnist_dataset_train, mnist_dataset_test

    check = SimpleFeatureContribution(per_class=False, random_state=42)

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.08, 0.005)}),
        'test': has_entries({'Brightness': close_to(0.234, 0.001)}),
        'train-test difference': has_entries({'Brightness': close_to(-0.153, 0.001)})
    }))


def test_no_drift_object_detection(coco_train_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    check = SimpleFeatureContribution(per_class=False, random_state=42)

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': equal_to(0)}),
        'test': has_entries({'Brightness': equal_to(0)}),
        'train-test difference': has_entries({'Brightness': equal_to(0)}),
    }))


def test_drift_object_detection(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    check = SimpleFeatureContribution(per_class=False, random_state=42)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias(train.batch_to_labels)

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.062, 0.01)}),
        'test': has_entries({'Brightness': equal_to(0)}),
        'train-test difference': has_entries({'Brightness': close_to(0.062, 0.01)}),
    }))


def test_no_drift_classification_per_class(mnist_dataset_train):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = SimpleFeatureContribution(per_class=True, random_state=42)

    # Act
    result = check.run(train, test, n_samples=None)

    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train':  has_entries({'1': equal_to(0)}),
                                   'test':  has_entries({'1': equal_to(0)}),
                                   'train-test difference':  has_entries({'1': equal_to(0)})}),
    }))


def test_drift_classification_per_class(mnist_dataset_train, mnist_dataset_test):
    mnist_dataset_test.batch_to_images = mnist_batch_to_images_with_bias

    train, test = mnist_dataset_train, mnist_dataset_test

    check = SimpleFeatureContribution(per_class=True, random_state=42)

    # Act
    result = check.run(train, test, n_samples=None)

    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train':  has_entries({'1': equal_to(0)}),
                                   'test':  has_entries({'1': close_to(0.659, 0.01)}),
                                   'train-test difference':  has_entries({'1': close_to(-0.659, 0.01)})}),
    }))


def test_no_drift_object_detection_per_class(coco_train_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    check = SimpleFeatureContribution(per_class=True, random_state=42)

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train':  has_entries({'clock': equal_to(0)}),
                                   'test':  has_entries({'clock': equal_to(0)}),
                                   'train-test difference':  has_entries({'clock': equal_to(0)})}),
    }))


def test_no_drift_object_detection_per_class_min_pps(coco_train_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    check = SimpleFeatureContribution(per_class=True, random_state=42, min_pps_to_show=2)

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train':  has_entries({'clock': equal_to(0)}),
                                   'test':  has_entries({'clock': equal_to(0)}),
                                   'train-test difference':  has_entries({'clock': equal_to(0)})}),
    }))
    assert_that(result.display, equal_to([]))


def test_drift_object_detections_min_pps(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    check = SimpleFeatureContribution(per_class=False, random_state=42, min_pps_to_show=2)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias(train.batch_to_labels)

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.062, 0.01)}),
        'test': has_entries({'Brightness': equal_to(0)}),
        'train-test difference': has_entries({'Brightness': close_to(0.062, 0.01)}),
    }))
    assert_that(result.display, equal_to([]))


def test_drift_object_detection_per_class(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    check = SimpleFeatureContribution(per_class=True, random_state=42)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias(train.batch_to_labels)

    # Act
    result = check.run(train, test)
    
    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train':  has_entries({'fork': equal_to(0)}),
                                   'test':  has_entries({'fork': close_to(0.0025, 0.001)}),
                                   'train-test difference':  has_entries({'fork': close_to(-0.0025, 0.001)})}),
    }))


def test_train_test_condition_pps_train_pass(coco_train_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    condition_value = 0.3
    check = SimpleFeatureContribution(per_class=False, random_state=42
                                      ).add_condition_feature_pps_in_train_not_greater_than(condition_value)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        name=f'Train properties\' Predictive Power Score is not greater than {condition_value}'
    ))


def test_train_test_condition_pps_train_fail(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    condition_value = 0.09
    check = SimpleFeatureContribution(per_class=False, random_state=42
                                      ).add_condition_feature_pps_in_train_not_greater_than(condition_value)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias(train.batch_to_labels)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train properties\' Predictive Power Score is not greater than {condition_value}',
        details=(
            'Features in train dataset with PPS above threshold: '
            '{\'Mean Red Relative Intensity\': \'0.11\', '
            '\'Aspect Ratio\': \'0.09\'}'
        )
    ))


def test_train_test_condition_pps_train_pass_per_class(mnist_dataset_train):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    condition_value = 0.3
    check = SimpleFeatureContribution(per_class=True, random_state=42
                                      ).add_condition_feature_pps_in_train_not_greater_than(condition_value)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        name=f'Train properties\' Predictive Power Score is not greater than {condition_value}'
    ))


def test_train_test_condition_pps_train_fail_per_class(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    condition_value = 0.3
    check = SimpleFeatureContribution(per_class=True, random_state=42
                                      ).add_condition_feature_pps_in_train_not_greater_than(condition_value)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias_one_class(train.batch_to_labels)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train properties\' Predictive Power Score is not greater than {condition_value}',
        details='Features in train dataset with PPS above threshold: {\'RMS Contrast\': \'0.83\', '
                '\'Brightness\': \'0.5\', \'Mean Blue Relative Intensity\': \'0.33\'}'
    ))
