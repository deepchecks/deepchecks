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
from hamcrest import (all_of, any_of, assert_that, calling, close_to, contains_exactly, equal_to, has_entries, has_key,
                      has_length, has_properties, instance_of, is_, raises)
from hamcrest.core.matcher import Matcher

from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksProcessError
from deepchecks.vision.checks import LabelPropertyOutliers
from deepchecks.vision.utils.label_prediction_properties import DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES


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


def test_outliers_check_coco(coco_visiondata_train):
    # Act
    result = LabelPropertyOutliers().run(coco_visiondata_train)

    # Assert
    assert_that(result, is_correct_label_property_outliers_result(DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES))
    assert_that(result.value, has_entries({
        'Number of Bounding Boxes Per Image': has_entries({
            'outliers_identifiers': contains_exactly('21', '30', '1', '5', '11', '20'),
            'lower_limit': is_(0),
            'upper_limit': is_(20.125)
        }),
        'Bounding Box Area (in pixels)': instance_of(dict),
    }))

def test_tf_coco_batch_without_boxes(tf_coco_visiondata_train):
    # Act
    result = LabelPropertyOutliers().run(tf_coco_visiondata_train)

    # Assert
    assert_that(result, is_correct_label_property_outliers_result(DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES))
    assert_that(result.value, has_entries({
        'Number of Bounding Boxes Per Image': has_entries({
            'lower_limit': close_to(1, 1),
        }),
        'Bounding Box Area (in pixels)': instance_of(dict),
    }))

def test_outliers_check_coco_segmentation(segmentation_coco_visiondata_train):
    # Act
    result = LabelPropertyOutliers().run(segmentation_coco_visiondata_train)

    # Assert
    assert_that(result.value, has_entries({
        'Number of Classes Per Image': has_entries({
            'outliers_identifiers': contains_exactly('3', '8'),
            'lower_limit': is_(2),
            'upper_limit': is_(2)
        }),
        'Segment Area (in pixels)': instance_of(dict),
    }))


def test_outliers_check_coco_without_display(coco_visiondata_train):
    # Act
    result = LabelPropertyOutliers().run(coco_visiondata_train, with_display=False)

    # Assert
    assert_that(result,
                is_correct_label_property_outliers_result(DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES, with_display=False))
    assert_that(result.value, has_entries({
        'Number of Bounding Boxes Per Image': has_entries({
            'outliers_identifiers': contains_exactly('21', '30', '1', '5', '11', '20'),
            'lower_limit': is_(0),
            'upper_limit': is_(20.125)
        }),
        'Bounding Box Area (in pixels)': instance_of(dict),
    }))


def test_property_outliers_check_mnist(mnist_visiondata_train):
    # Arrange
    properties = [{
        'name': 'test',
        'method': lambda labels: labels,
        'output_type': 'categorical'
    }]
    check = LabelPropertyOutliers(label_properties=properties)
    # Act
    result = check.run(mnist_visiondata_train)

    # Assert
    assert_that(result, is_correct_label_property_outliers_result(properties))
    assert_that(result.value, has_entries({
        'test': has_entries({
            'outliers_identifiers': has_length(0),
            'lower_limit': is_(0),
            'upper_limit': is_(9)
        })
    }))


def test_run_on_data_with_only_labels(coco_test_only_labels):
    # Act - Assert check runs without exception
    result = LabelPropertyOutliers().run(coco_test_only_labels)
    # Assert
    assert_that(result, is_correct_label_property_outliers_result(DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES))


def test_exception_on_custom_task(mnist_train_custom_task):
    # Arrange
    check = LabelPropertyOutliers()

    # Act & Assert exception
    assert_that(calling(check.run).with_args(mnist_train_custom_task),
                raises(DeepchecksProcessError, 'task type TaskType.OTHER does not have default label properties '
                                               'defined'))


def test_not_enough_samples_for_iqr(coco_train_very_small):
    # Arrange
    check = LabelPropertyOutliers()
    # Act
    result = check.run(coco_train_very_small)
    # Assert
    assert_that(result, is_correct_label_property_outliers_result(DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES))
    assert_that(result.value, has_entries({
        'Number of Bounding Boxes Per Image': equal_to('Not enough non-null samples to calculate outliers.'),
        'Bounding Box Area (in pixels)': instance_of(dict),
    }))


def test_string_property_exception(mnist_visiondata_train):
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
    assert_that(calling(check.run).with_args(mnist_visiondata_train),
                raises(DeepchecksProcessError, 'For outliers, properties are expected to be only numeric types but '
                                               'found non-numeric value for property test'))


def test_incorrect_properties_count_exception(mnist_visiondata_train):
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
    assert_that(calling(check.run).with_args(mnist_visiondata_train),
                raises(DeepchecksProcessError, 'Properties are expected to return value per image but instead got 65 '
                                               'values for 64 images for property test'))

