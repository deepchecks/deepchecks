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
"""Test for the check property label correlation change."""
from copy import copy

import numpy as np
import pandas as pd
from hamcrest import assert_that, close_to, equal_to, greater_than, has_entries, has_length

from deepchecks.vision.checks import PropertyLabelCorrelationChange
from deepchecks.vision.utils.transformations import un_normalize_batch
from tests.base.utils import equal_condition_result
from tests.vision.vision_conftest import *


def mnist_batch_to_images_with_bias(batch):
    """Create function which inverse the data normalization."""
    tensor = batch[0]
    tensor = tensor.permute(0, 2, 3, 1)
    ret = un_normalize_batch(tensor, (0.1307,), (0.3081,))
    for i, label in enumerate(batch[1]):
        label = label.cpu().detach()
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


def test_no_drift_classification(mnist_dataset_train, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = PropertyLabelCorrelationChange(per_class=False, random_state=42)

    # Act
    result = check.run(train, test, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.08, 0.005)}),
        'test': has_entries({'Brightness': close_to(0.08, 0.005)}),
        'train-test difference': has_entries({'Brightness': equal_to(0)})
    }))


def test_drift_classification(mnist_dataset_train, mnist_dataset_test, device):
    mnist_dataset_test.batch_to_images = mnist_batch_to_images_with_bias

    train, test = mnist_dataset_train, mnist_dataset_test

    check = PropertyLabelCorrelationChange(per_class=False, random_state=42)

    # Act
    result = check.run(train, test, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.08, 0.005)}),
        'test': has_entries({'Brightness': close_to(0.234, 0.001)}),
        'train-test difference': has_entries({'Brightness': close_to(-0.153, 0.001)})
    }))


def test_no_drift_object_detection(coco_train_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    check = PropertyLabelCorrelationChange(per_class=False, random_state=42)

    # Act
    result = check.run(train, test, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': equal_to(0)}),
        'test': has_entries({'Brightness': equal_to(0)}),
        'train-test difference': has_entries({'Brightness': equal_to(0)}),
    }))


def test_drift_object_detection(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    check = PropertyLabelCorrelationChange(per_class=False, random_state=42)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias(train.batch_to_labels)

    # Act
    result = check.run(train, test, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.062, 0.01)}),
        'test': has_entries({'Brightness': equal_to(0)}),
        'train-test difference': has_entries({'Brightness': close_to(0.062, 0.01)}),
    }))
    assert_that(result.display, has_length(greater_than(0)))


def test_drift_object_detection_without_display(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    check = PropertyLabelCorrelationChange(per_class=False, random_state=42)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias(train.batch_to_labels)

    # Act
    result = check.run(train, test, device=device, with_display=False)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.062, 0.01)}),
        'test': has_entries({'Brightness': equal_to(0)}),
        'train-test difference': has_entries({'Brightness': close_to(0.062, 0.01)}),
    }))
    assert_that(result.display, has_length(0))


def test_no_drift_classification_per_class(mnist_dataset_train, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = PropertyLabelCorrelationChange(per_class=True, random_state=42)

    # Act
    result = check.run(train, test, n_samples=None, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train':  has_entries({'1': equal_to(0)}),
                                   'test':  has_entries({'1': equal_to(0)}),
                                   'train-test difference':  has_entries({'1': equal_to(0)})}),
    }))


def test_drift_classification_per_class(mnist_dataset_train, mnist_dataset_test, device):
    mnist_dataset_test.batch_to_images = mnist_batch_to_images_with_bias

    train, test = mnist_dataset_train, mnist_dataset_test

    check = PropertyLabelCorrelationChange(per_class=True, random_state=42)

    # Act
    result = check.run(train, test, n_samples=None, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train':  has_entries({'1': equal_to(0)}),
                                   'test':  has_entries({'1': close_to(0.659, 0.01)}),
                                   'train-test difference':  has_entries({'1': close_to(-0.659, 0.01)})}),
    }))


def test_no_drift_object_detection_per_class(coco_train_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    check = PropertyLabelCorrelationChange(per_class=True, random_state=42)

    # Act
    result = check.run(train, test, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train':  has_entries({'clock': equal_to(0)}),
                                   'test':  has_entries({'clock': equal_to(0)}),
                                   'train-test difference':  has_entries({'clock': equal_to(0)})}),
    }))


def test_no_drift_object_detection_per_class_min_pps(coco_train_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    check = PropertyLabelCorrelationChange(per_class=True, random_state=42, min_pps_to_show=2)

    # Act
    result = check.run(train, test, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train':  has_entries({'clock': equal_to(0)}),
                                   'test':  has_entries({'clock': equal_to(0)}),
                                   'train-test difference':  has_entries({'clock': equal_to(0)})}),
    }))
    assert_that(result.display, equal_to([]))


def test_drift_object_detections_min_pps(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    check = PropertyLabelCorrelationChange(per_class=False, random_state=42, min_pps_to_show=2)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias(train.batch_to_labels)

    # Act
    result = check.run(train, test, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.062, 0.01)}),
        'test': has_entries({'Brightness': equal_to(0)}),
        'train-test difference': has_entries({'Brightness': close_to(0.062, 0.01)}),
    }))
    assert_that(result.display, equal_to([]))


def test_drift_object_detection_per_class(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    check = PropertyLabelCorrelationChange(per_class=True, random_state=42)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias(train.batch_to_labels)

    # Act
    result = check.run(train, test, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train':  has_entries({'fork': equal_to(0)}),
                                   'test':  has_entries({'fork': close_to(0.0025, 0.001)}),
                                   'train-test difference':  has_entries({'fork': close_to(-0.0025, 0.001)})}),
    }))


def test_train_test_condition_pps_train_pass(coco_train_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    condition_value = 0.3
    check = PropertyLabelCorrelationChange(per_class=False, random_state=42
                                          ).add_condition_property_pps_in_train_less_than(condition_value)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test, device=device)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='0 PPS found for all properties in train dataset',
        name=f'Train properties\' Predictive Power Score is less than {condition_value}'
    ))


def test_train_test_condition_pps_train_fail(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    condition_value = 0.1
    check = PropertyLabelCorrelationChange(per_class=False, random_state=42
                                          ).add_condition_property_pps_in_train_less_than(condition_value)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias(train.batch_to_labels)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test, device=device)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train properties\' Predictive Power Score is less than {condition_value}',
        details=(
            'Properties in train dataset with PPS above threshold: '
            '{\'Mean Red Relative Intensity\': \'0.11\'}'
        )
    ))


def test_train_test_condition_pps_train_pass_per_class(mnist_dataset_train, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    condition_value = 0.3
    check = PropertyLabelCorrelationChange(per_class=True, random_state=42
                                          ).add_condition_property_pps_in_train_less_than(condition_value)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test, device=device)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Found highest PPS in train dataset 0.06 for property Brightness and class 1',
        name=f'Train properties\' Predictive Power Score is less than {condition_value}'
    ))


def test_train_test_condition_pps_train_fail_per_class(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    condition_value = 0.3
    check = PropertyLabelCorrelationChange(per_class=True, random_state=42
                                          ).add_condition_property_pps_in_train_less_than(condition_value)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias_one_class(train.batch_to_labels)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test, device=device)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train properties\' Predictive Power Score is less than {condition_value}',
        details='Properties and classes in train dataset with PPS above threshold: {\'RMS Contrast\': {\'clock\': '
                '\'0.83\'}, \'Brightness\': {\'clock\': \'0.5\', \'teddy bear\': \'0.5\'}, '
                '\'Mean Blue Relative Intensity\': {\'clock\': \'0.33\'}}'
    ))


def test_train_test_condition_pps_diff_pass(coco_train_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_train_visiondata
    condition_value = 0.01
    check = PropertyLabelCorrelationChange(per_class=False, random_state=42
                                          ).add_condition_property_pps_difference_less_than(condition_value)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test, device=device)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='0 PPS found for all properties',
        name=f'Train-Test properties\' Predictive Power Score difference is less than {condition_value}'
    ))


def test_train_test_condition_pps_positive_diff_fail(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    condition_value = 0.09
    check = PropertyLabelCorrelationChange(per_class=False, random_state=42
                                          ).add_condition_property_pps_difference_less_than(condition_value)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias(train.batch_to_labels)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test, device=device)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train-Test properties\' Predictive Power Score difference is less than {condition_value}',
        details=(
            'Properties with PPS difference above threshold: '
            '{\'Mean Red Relative Intensity\': \'0.1\'}'
        )
    ))


def test_train_test_condition_pps_diff_fail(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    condition_value = 0.09
    check = PropertyLabelCorrelationChange(per_class=False, random_state=42).\
        add_condition_property_pps_difference_less_than(condition_value, include_negative_diff=False)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias(train.batch_to_labels)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test, device=device)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train-Test properties\' Predictive Power Score difference is less than {condition_value}',
        details=(
            'Properties with PPS difference above threshold: '
            '{\'Mean Red Relative Intensity\': \'0.1\'}'
        )
    ))


def test_train_test_condition_pps_diff_pass_per_class(mnist_dataset_train, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    condition_value = 0.3
    check = PropertyLabelCorrelationChange(per_class=True, random_state=42
                                          ).add_condition_property_pps_difference_less_than(condition_value)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test, device=device)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        details='0 PPS found for all properties',
        is_pass=True,
        name=f'Train-Test properties\' Predictive Power Score difference is less than {condition_value}'
    ))


def test_train_test_condition_pps_positive_diff_fail_per_class(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    condition_value = 0.4
    check = PropertyLabelCorrelationChange(per_class=True, random_state=42).\
        add_condition_property_pps_difference_less_than(condition_value, include_negative_diff=False)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias_one_class(train.batch_to_labels)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test, device=device)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train-Test properties\' Predictive Power Score difference is less than {condition_value}',
        details='Properties and classes with PPS difference above threshold: {\'RMS Contrast\': {\'clock\': \'0.83\'}, '
                '\'Brightness\': {\'clock\': \'0.5\', \'teddy bear\': \'0.5\'}}'
    ))


def test_train_test_condition_pps_diff_fail_per_class(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    train, test = coco_train_visiondata, coco_test_visiondata
    condition_value = 0.3
    check = PropertyLabelCorrelationChange(per_class=True, random_state=42
                                          ).add_condition_property_pps_difference_less_than(condition_value)
    train = copy(train)
    train.batch_to_images = get_coco_batch_to_images_with_bias_one_class(train.batch_to_labels)

    # Act
    result = check.run(train_dataset=train,
                       test_dataset=test, device=device)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train-Test properties\' Predictive Power Score difference is less than {condition_value}',
        details='Properties and classes with PPS difference above threshold: {\'RMS Contrast\': {\'clock\': \'0.83\'}, '
                '\'Brightness\': {\'clock\': \'0.5\', \'teddy bear\': \'0.5\'}, \'Mean Blue Relative Intensity\': '
                '{\'clock\': \'0.33\'}}'
    ))


