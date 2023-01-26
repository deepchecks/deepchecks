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
"""Image Property Drift check tests"""
from hamcrest import (all_of, assert_that, close_to, greater_than, has_entries, has_items, has_key, has_length,
                      has_properties, instance_of)

from deepchecks.core import CheckResult
from deepchecks.vision.checks import ImagePropertyDrift
from deepchecks.vision.utils.image_properties import default_image_properties
from tests.base.utils import equal_condition_result


def test_image_property_drift_check(coco_visiondata_train, coco_visiondata_test):
    # Run
    result = ImagePropertyDrift().run(coco_visiondata_train, coco_visiondata_test)

    # Assert
    assert_that(result, is_correct_image_property_drift_result())

    assert_that(result.value, has_entries(
        {'Brightness': close_to(0.07, 0.01)}
    ))

    assert_that(result.reduce_output(), has_entries(
        {'Max Drift Score': close_to(0.07, 0.01)}
    ))


def test_image_property_drift_check_without_display(coco_visiondata_train, coco_visiondata_test):
    # Run
    result = ImagePropertyDrift(aggregation_method='mean').run(coco_visiondata_train, coco_visiondata_test,
                                                               with_display=False)

    # Assert
    assert_that(result, is_correct_image_property_drift_result(with_display=False))

    assert_that(result.value, has_entries(
        {'Brightness': close_to(0.07, 0.01)}
    ))

    assert_that(result.reduce_output(), has_entries(
        {'Mean Drift Score': close_to(0.05, 0.01)}
    ))


def test_image_property_drift_check_without_display_none_aggragation(coco_visiondata_train, coco_visiondata_test):
    # Run
    result = ImagePropertyDrift(aggregation_method='none').run(coco_visiondata_train, coco_visiondata_test,
                                                               with_display=False)

    # Assert
    assert_that(result, is_correct_image_property_drift_result(with_display=False))

    assert_that(result.value, has_entries(
        {'Brightness': close_to(0.07, 0.01)}
    ))

    assert_that(result.reduce_output(), has_entries(
        {'Brightness': close_to(0.07, 0.01)}
    ))


# def test_image_property_drift_check_limit_classes(coco_visiondata_train, coco_visiondata_test):
#     # Run
#     result = ImagePropertyDrift(classes_to_display=['bicycle', 'bench', 'bus', 'truck'], min_samples=5
#                                 ).run(coco_visiondata_train, coco_visiondata_test)
#
#     # Assert
#     assert_that(result, is_correct_image_property_drift_result())
#
#     assert_that(result.value, has_entries(
#         {'Brightness': close_to(0.13, 0.01)}
#     ))


# def test_image_property_drift_check_limit_classes_illegal(coco_visiondata_train, coco_visiondata_test):
#     check = ImagePropertyDrift(classes_to_display=['phone'])
#     assert_that(
#         calling(check.run).with_args(coco_visiondata_train, coco_visiondata_test),
#         raises(DeepchecksValueError, r'Provided list of class ids to display \[\'phone\'\] not found in training '
#                                      r'dataset.')
#     )


def test_image_property_drift_condition(coco_visiondata_train, coco_visiondata_test):
    result = ImagePropertyDrift().add_condition_drift_score_less_than().run(coco_visiondata_train, coco_visiondata_test)

    assert_that(result, is_correct_image_property_drift_result())
    assert_that(result.conditions_results, has_items(
        equal_condition_result(is_pass=True,
                               details='Found property Brightness with largest Earth Mover\'s Distance score 0.07',
                               name='Earth Mover\'s Distance < 0.1 for image properties drift'))
                )


def test_image_property_drift_fail_condition(coco_visiondata_train, coco_visiondata_test):
    result = (
        ImagePropertyDrift()
        .add_condition_drift_score_less_than(0.06)
        .run(coco_visiondata_train, coco_visiondata_test)
    )

    assert_that(result, is_correct_image_property_drift_result())
    assert_that(result.conditions_results, has_items(
        equal_condition_result(is_pass=False,
                               details='Earth Mover\'s Distance is above the threshold for the next properties:\n'
                                       'Aspect Ratio=0.07;\nBrightness=0.07;\nMean Green Relative Intensity=0.06',
                               name='Earth Mover\'s Distance < 0.06 for image properties drift'))
                )


def is_correct_image_property_drift_result(with_display: bool = True):
    value_assertion = all_of(
        instance_of(dict),
        *[has_key(single_property['name']) for single_property in default_image_properties])

    if with_display:
        display_assertion = all_of(
            instance_of(list),
            has_length(greater_than(1)),
            # TODO
        )
    else:
        display_assertion = all_of(
            instance_of(list),
            has_length(0),
            # TODO
        )
    return all_of(
        instance_of(CheckResult),
        has_properties({
            'value': value_assertion,
            'header': 'Image Property Drift',
            'display': display_assertion
        })
    )


def test_run_on_data_with_only_images(mnist_train_only_images, mnist_test_only_images):
    # Act - Assert check runs without exception
    ImagePropertyDrift().run(mnist_train_only_images, mnist_test_only_images)
