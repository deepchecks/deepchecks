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
from hamcrest import (all_of, any_of, assert_that, calling, contains_exactly, empty, equal_to, has_entries, has_key,
                      has_length, has_properties, instance_of, is_, raises)
from hamcrest.core.matcher import Matcher

from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksProcessError
from deepchecks.vision.checks import LabelPropertyOutliers
from deepchecks.vision.utils.label_prediction_properties import DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES
from tests.vision.vision_conftest import *


def is_correct_label_property_outliers_result(props, with_display: bool = True) -> Matcher:
    props = [p for p in props if p['output_type'] != 'class_id']
    value_assertion = all_of(
        instance_of(dict),
        *[has_key(single_property['name']) for single_property in props])

    if with_display:
        display_assertion = all_of(
            instance_of(list),
            any_of(has_length(1), has_length(2), has_length(3), has_length(4)),
        )
    else:
        display_assertion = all_of(
            instance_of(list),
            has_length(0),
        )

    return all_of(
        instance_of(CheckResult),
        has_properties({
            'value': value_assertion,
            'display': display_assertion
        })
    )


def test_outliers_check_coco(coco_train_visiondata, device):
    # Act
    result = LabelPropertyOutliers().run(coco_train_visiondata, device=device)

    # Assert
    assert_that(result, is_correct_label_property_outliers_result(DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES))
    assert_that(result.value, has_entries({
        'Number of Bounding Boxes Per Image': has_entries({
            'indices': contains_exactly(30, 21, 43, 52, 33, 37),
            'lower_limit': is_(0),
            'upper_limit': is_(20.125)
        }),
        'Bounding Box Area (in pixels)': instance_of(dict),
    }))


def test_outliers_check_coco_segmentation(segmentation_coco_train_visiondata, device):
    # Act
    result = LabelPropertyOutliers().run(segmentation_coco_train_visiondata, device=device)

    # Assert
    assert_that(result.value, has_entries({
        'Number of Classes Per Image': has_entries({
            'indices': [8, 3],
            'lower_limit': is_(2),
            'upper_limit': is_(2)
        }),
        'Segment Area (in pixels)': instance_of(dict),
    }))


def test_outliers_check_coco_without_display(coco_train_visiondata, device):
    # Act
    result = LabelPropertyOutliers().run(coco_train_visiondata, device=device, with_display=False)

    # Assert
    assert_that(result,
                is_correct_label_property_outliers_result(DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES, with_display=False))
    assert_that(result.value, has_entries({
        'Number of Bounding Boxes Per Image': has_entries({
            'indices': contains_exactly(30, 21, 43, 52, 33, 37),
            'lower_limit': is_(0),
            'upper_limit': is_(20.125)
        }),
        'Bounding Box Area (in pixels)': instance_of(dict),
    }))


def test_property_outliers_check_mnist(mnist_dataset_train, device):
    # Arrange
    properties = [{
        'name': 'test',
        'method': lambda labels: labels.tolist(),
        'output_type': 'categorical'
    }]
    check = LabelPropertyOutliers(label_properties=properties)
    # Act
    result = check.run(mnist_dataset_train, device=device, n_samples=None)

    # Assert
    assert_that(result, is_correct_label_property_outliers_result(properties))
    assert_that(result.value, has_entries({
        'test': has_entries({
            'indices': has_length(0),
            'lower_limit': is_(0),
            'upper_limit': is_(9)
        })
    }))


def test_run_on_data_with_only_labels(coco_train_visiondata, device):
    # Arrange
    data = coco_train_visiondata.copy(random_state=0)
    data._image_formatter_error = 'fake error'
    # Act - Assert check runs without exception
    result = LabelPropertyOutliers().run(data, device=device)
    # Assert
    assert_that(result, is_correct_label_property_outliers_result(DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES))


def test_exception_on_custom_task(coco_train_custom_task, device):
    # Arrange
    check = LabelPropertyOutliers()

    # Act & Assert exception
    assert_that(calling(check.run).with_args(coco_train_custom_task, device=device),
                raises(DeepchecksProcessError, 'task type TaskType.OTHER does not have default label properties '
                                               'defined'))


def test_run_on_custom_task_with_custom_properties(coco_train_custom_task, device):
    # Arrange
    def custom_property(labels):
        return [1] * len(labels)
    properties = [{
        'name': 'test',
        'method': custom_property,
        'output_type': 'categorical'
    }]

    # Act - Assert check runs without exception
    result = LabelPropertyOutliers(label_properties=properties).run(coco_train_custom_task, device=device)
    # Assert
    assert_that(result, is_correct_label_property_outliers_result(properties))


def test_not_enough_samples_for_iqr(coco_train_visiondata, device):
    # Arrange
    five_samples_mnist = coco_train_visiondata.copy(n_samples=5, random_state=0)
    check = LabelPropertyOutliers()
    # Act
    result = check.run(five_samples_mnist, device=device)
    # Assert
    assert_that(result, is_correct_label_property_outliers_result(DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES))
    assert_that(result.value, has_entries({
        'Number of Bounding Boxes Per Image': equal_to('Not enough non-null samples to calculate outliers.'),
        'Bounding Box Area (in pixels)': instance_of(dict),
    }))


def test_string_property_exception(mnist_dataset_train, device):
    # Arrange
    def string_property(labels):
        return ['test'] * len(labels)
    image_properties = [{
        'name': 'test',
        'method': string_property,
        'output_type': 'categorical'
    }]
    check = LabelPropertyOutliers(label_properties=image_properties)
    # Act - Assert check raise exception
    assert_that(calling(check.run).with_args(mnist_dataset_train, device=device),
                raises(DeepchecksProcessError, 'For outliers, properties are expected to be only numeric types but '
                                               'found non-numeric value for property test'))


def test_incorrect_properties_count_exception(mnist_dataset_train, device):
    # Arrange
    def too_many_property(labels):
        return ['test'] * (len(labels) + 1)
    image_properties = [{
        'name': 'test',
        'method': too_many_property,
        'output_type': 'categorical'
    }]
    check = LabelPropertyOutliers(label_properties=image_properties)
    # Act - Assert check raise exception
    assert_that(calling(check.run).with_args(mnist_dataset_train, device=device),
                raises(DeepchecksProcessError, 'Properties are expected to return value per image but instead got 65 '
                                               'values for 64 images for property test'))
