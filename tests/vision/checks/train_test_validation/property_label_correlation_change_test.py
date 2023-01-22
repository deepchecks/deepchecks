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

import numpy as np
import torch
from hamcrest import assert_that, close_to, equal_to, greater_than, has_entries, has_length

from deepchecks.vision.checks import PropertyLabelCorrelationChange
from deepchecks.vision.utils.test_utils import replace_collate_fn_visiondata
from tests.base.utils import equal_condition_result


def coco_collate_with_bias_one_class(data):
    raw_images = [x[0] for x in data]
    images = [np.array(x) for x in raw_images]

    def move_class(tensor):
        return torch.index_select(tensor, 1, torch.LongTensor([4, 0, 1, 2, 3]).to(tensor.device)) \
            if len(tensor) > 0 else tensor

    labels = [move_class(x[1]) for x in data]
    for i, bboxes_per_image in enumerate(labels):
        for bbox in bboxes_per_image:
            if bbox[0] == 74:
                x, y, w, h = [round(float(n)) for n in bbox[1:]]
                images[i][y:y + h, x:x + w] = images[i][y:y + h, x:x + w].clip(min=200)
    return {'images': images, 'labels': labels}


def test_no_drift_classification(mnist_visiondata_train):
    # Arrange
    train, test = mnist_visiondata_train, mnist_visiondata_train
    check = PropertyLabelCorrelationChange(per_class=False)

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.08, 0.005)}),
        'test': has_entries({'Brightness': close_to(0.08, 0.005)}),
        'train-test difference': has_entries({'Brightness': equal_to(0)})
    }))


def test_drift_classification(mnist_train_brightness_bias, mnist_visiondata_test):
    # Arrange
    check = PropertyLabelCorrelationChange(per_class=False)

    # Act
    result = check.run(mnist_train_brightness_bias, mnist_visiondata_test)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.248, 0.005)}),
        'test': has_entries({'Brightness': close_to(0.02, 0.001)}),
        'train-test difference': has_entries({'Brightness': close_to(0.228, 0.001)})
    }))


def test_no_drift_object_detection(coco_visiondata_train):
    # Arrange
    check = PropertyLabelCorrelationChange(per_class=False)

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_train)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': equal_to(0)}),
        'test': has_entries({'Brightness': equal_to(0)}),
        'train-test difference': has_entries({'Brightness': equal_to(0)}),
    }))


def test_drift_object_detection(coco_train_brightness_bias, coco_visiondata_test):
    # Arrange
    check = PropertyLabelCorrelationChange(per_class=False)

    # Act
    result = check.run(coco_train_brightness_bias, coco_visiondata_test)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.087, 0.01)}),
        'test': has_entries({'Brightness': close_to(0.016, 0.01)}),
        'train-test difference': has_entries({'Brightness': close_to(0.071, 0.01)}),
    }))
    assert_that(result.display, has_length(greater_than(0)))


## TODO: make this test work (its failing sometime in random)
# def test_drift_object_detection_tf(tf_coco_visiondata_train, tf_coco_visiondata_test):
#     # Arrange
#     check = PropertyLabelCorrelationChange(per_class=False)
#
#     # Act
#     result = check.run(tf_coco_visiondata_train, tf_coco_visiondata_test)
#
#     # Assert
#     assert_that(result.value, has_entries({
#         'train': has_entries({'Aspect Ratio': close_to(0.067, 0.01)}),
#         'test': has_entries({'Aspect Ratio': close_to(0.048, 0.01)}),
#         'train-test difference': has_entries({'Aspect Ratio': close_to(0.018, 0.01)}),
#     }))
#     assert_that(result.display, has_length(greater_than(0)))


def test_drift_object_detection_without_display(coco_train_brightness_bias, coco_visiondata_test):
    # Arrange
    check = PropertyLabelCorrelationChange(per_class=False)

    # Act
    result = check.run(coco_train_brightness_bias, coco_visiondata_test, with_display=False)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.087, 0.01)}),
        'test': has_entries({'Brightness': close_to(0.016, 0.01)}),
        'train-test difference': has_entries({'Brightness': close_to(0.071, 0.01)}),
    }))
    assert_that(result.display, has_length(0))


def test_no_drift_classification_per_class(mnist_visiondata_train):
    # Arrange
    check = PropertyLabelCorrelationChange(per_class=True)

    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_train)

    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train': has_entries({'1': equal_to(0)}),
                                   'test': has_entries({'1': equal_to(0)}),
                                   'train-test difference': has_entries({'1': equal_to(0)})}),
    }))


def test_drift_classification_per_class(mnist_train_brightness_bias, mnist_visiondata_test):
    # Arrange
    check = PropertyLabelCorrelationChange(per_class=True)

    # Act
    result = check.run(mnist_train_brightness_bias, mnist_visiondata_test)

    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train': has_entries({'1': close_to(0.807, 0.01)}),
                                   'test': has_entries({'1': close_to(0.0, 0.01)}),
                                   'train-test difference': has_entries({'1': close_to(0.807, 0.01)})}),
    }))


def test_no_drift_object_detection_per_class_min_pps(coco_visiondata_train):
    # Arrange
    check = PropertyLabelCorrelationChange(per_class=True, min_pps_to_show=2)

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_train)

    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train': has_entries({'clock': equal_to(0)}),
                                   'test': has_entries({'clock': equal_to(0)}),
                                   'train-test difference': has_entries({'clock': equal_to(0)})}),
    }))
    assert_that(result.display, equal_to([]))


def test_drift_object_detections_min_pps(coco_train_brightness_bias, coco_visiondata_test):
    # Arrange
    check = PropertyLabelCorrelationChange(per_class=False, min_pps_to_show=2)

    # Act
    result = check.run(coco_train_brightness_bias, coco_visiondata_test)

    # Assert
    assert_that(result.value, has_entries({
        'train': has_entries({'Brightness': close_to(0.087, 0.01)}),
        'test': has_entries({'Brightness': close_to(0.016, 0.01)}),
        'train-test difference': has_entries({'Brightness': close_to(0.071, 0.01)}),
    }))
    assert_that(result.display, equal_to([]))


def test_drift_object_detection_per_class(coco_train_brightness_bias, coco_visiondata_test):
    # Arrange
    check = PropertyLabelCorrelationChange(per_class=True)

    # Act
    result = check.run(coco_train_brightness_bias, coco_visiondata_test)

    # Assert
    assert_that(result.value, has_entries({
        'Brightness': has_entries({'train': has_entries({'bed': equal_to(0)}),
                                   'test': has_entries({'bed': close_to(0.0025, 0.001)}),
                                   'train-test difference': has_entries({'bed': close_to(-0.0025, 0.001)})}),
    }))


def test_train_test_condition_pps_train_pass(coco_visiondata_train):
    # Arrange
    condition_value = 0.3
    check = PropertyLabelCorrelationChange(per_class=False) \
        .add_condition_property_pps_in_train_less_than(condition_value)

    # Act
    result = check.run(train_dataset=coco_visiondata_train, test_dataset=coco_visiondata_train)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Found highest PPS in train dataset 0.02 for property Mean Red Relative Intensity',
        name=f'Train properties\' Predictive Power Score is less than {condition_value}'
    ))


def test_train_test_condition_pps_train_fail(coco_train_brightness_bias, coco_visiondata_test):
    # Arrange
    condition_value = 0.10
    check = PropertyLabelCorrelationChange(per_class=False) \
        .add_condition_property_pps_in_train_less_than(condition_value)
    # Act
    result = check.run(train_dataset=coco_train_brightness_bias, test_dataset=coco_visiondata_test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train properties\' Predictive Power Score is less than {condition_value}',
        details=(
            'Properties in train dataset with PPS above threshold: '
            '{\'RMS Contrast\': \'0.11\'}'
        )
    ))


def test_train_test_condition_pps_train_pass_per_class(mnist_visiondata_train):
    # Arrange
    train, test = mnist_visiondata_train, mnist_visiondata_train
    condition_value = 0.3
    check = PropertyLabelCorrelationChange(per_class=True) \
        .add_condition_property_pps_in_train_less_than(condition_value)

    # Act
    result = check.run(train_dataset=train, test_dataset=test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='0 PPS found for all properties in train dataset',
        name=f'Train properties\' Predictive Power Score is less than {condition_value}'
    ))


def test_train_test_condition_pps_train_fail_per_class(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    train = replace_collate_fn_visiondata(coco_visiondata_train, coco_collate_with_bias_one_class)
    condition_value = 0.3
    check = PropertyLabelCorrelationChange(per_class=True) \
        .add_condition_property_pps_in_train_less_than(condition_value)

    # Act
    result = check.run(train_dataset=train, test_dataset=coco_visiondata_test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train properties\' Predictive Power Score is less than {condition_value}',
        details='Properties and classes in train dataset with PPS above threshold: {'
                '\'RMS Contrast\': {\'clock\': \'0.83\'}}'
    ))


def test_train_test_condition_pps_diff_pass(coco_visiondata_train):
    # Arrange
    train, test = coco_visiondata_train, coco_visiondata_train
    condition_value = 0.03
    check = PropertyLabelCorrelationChange(per_class=False) \
        .add_condition_property_pps_difference_less_than(condition_value)

    # Act
    result = check.run(train_dataset=train, test_dataset=test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Found highest PPS 0.02 for property Mean Red Relative Intensity',
        name=f'Train-Test properties\' Predictive Power Score difference is less than {condition_value}'
    ))


def test_train_test_condition_pps_positive_diff_fail(coco_train_brightness_bias, coco_visiondata_test):
    # Arrange
    condition_value = 0.08
    check = PropertyLabelCorrelationChange(per_class=False
                                           ).add_condition_property_pps_difference_less_than(condition_value)

    # Act
    result = check.run(train_dataset=coco_train_brightness_bias, test_dataset=coco_visiondata_test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train-Test properties\' Predictive Power Score difference is less than {condition_value}',
        details=(
            'Properties with PPS difference above threshold: '
            '{\'Mean Red Relative Intensity\': \'0.09\', \'RMS Contrast\': \'0.11\'}'
        )
    ))


def test_train_test_condition_pps_diff_fail(coco_train_brightness_bias, coco_visiondata_test):
    # Arrange
    condition_value = 0.08
    check = PropertyLabelCorrelationChange(per_class=False). \
        add_condition_property_pps_difference_less_than(condition_value, include_negative_diff=False)
    # Act
    result = check.run(train_dataset=coco_train_brightness_bias, test_dataset=coco_visiondata_test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train-Test properties\' Predictive Power Score difference is less than {condition_value}',
        details=(
            'Properties with PPS difference above threshold: '
            '{\'Mean Red Relative Intensity\': \'0.09\', \'RMS Contrast\': \'0.11\'}'
        )
    ))


def test_train_test_condition_pps_diff_pass_per_class(mnist_visiondata_train):
    # Arrange
    train, test = mnist_visiondata_train, mnist_visiondata_train
    condition_value = 0.3
    check = PropertyLabelCorrelationChange(per_class=True, random_state=42
                                           ).add_condition_property_pps_difference_less_than(condition_value)

    # Act
    result = check.run(train_dataset=train, test_dataset=test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        details='0 PPS found for all properties',
        is_pass=True,
        name=f'Train-Test properties\' Predictive Power Score difference is less than {condition_value}'
    ))


def test_train_test_condition_pps_positive_diff_fail_per_class(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    train = replace_collate_fn_visiondata(coco_visiondata_train, coco_collate_with_bias_one_class)
    condition_value = 0.4
    check = PropertyLabelCorrelationChange(per_class=True, random_state=42). \
        add_condition_property_pps_difference_less_than(condition_value, include_negative_diff=False)

    # Act
    result = check.run(train_dataset=train, test_dataset=coco_visiondata_test)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train-Test properties\' Predictive Power Score difference is less than {condition_value}',
        details='Properties and classes with PPS difference above threshold: '
                '{\'RMS Contrast\': {\'clock\': \'0.83\'}}'
    ))
