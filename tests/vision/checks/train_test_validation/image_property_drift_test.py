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
from hamcrest import (all_of, assert_that, calling, close_to, contains_exactly,
                      equal_to, greater_than, has_entries, has_key, has_length,
                      has_properties, has_property, instance_of,
                      matches_regexp, raises)

from deepchecks.core import CheckResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import (DeepchecksNotImplementedError,
                                    DeepchecksValueError)
from deepchecks.vision.checks import ImagePropertyDrift
from deepchecks.vision.datasets.classification.mnist import MNISTData
from deepchecks.vision.utils.image_properties import default_image_properties


def test_image_property_drift_check(coco_train_visiondata, coco_test_visiondata, device):
    # Run
    result = ImagePropertyDrift().run(coco_train_visiondata, coco_test_visiondata, device=device)

    # Assert
    assert_that(result, is_correct_image_property_drift_result())

    assert_that(result.value, has_entries(
        {'Brightness': close_to(0.07, 0.01)}
    ))


def test_image_property_drift_check_limit_classes(coco_train_visiondata, coco_test_visiondata, device):
    # Run
    result = ImagePropertyDrift(classes_to_display=['bicycle', 'bench', 'bus', 'truck'], min_samples=5
                                ).run(coco_train_visiondata, coco_test_visiondata, device=device)

    # Assert
    assert_that(result, is_correct_image_property_drift_result())

    assert_that(result.value, has_entries(
        {'Brightness': close_to(0.13, 0.01)}
    ))


def test_image_property_drift_check_limit_classes_illegal(coco_train_visiondata, coco_test_visiondata, device):
    check = ImagePropertyDrift(classes_to_display=['phone'])
    assert_that(
        calling(check.run).with_args(coco_train_visiondata, coco_test_visiondata, device=device),
        raises(DeepchecksValueError,  r'Provided list of class ids to display \[\'phone\'\] not found in training '
                                      r'dataset.')
    )


def test_image_property_drift_initialization_with_empty_list_of_image_properties():
    assert_that(
        calling(ImagePropertyDrift).with_args(image_properties=[]),
        raises(DeepchecksValueError, 'Properties list can\'t be empty')
    )


def test_image_property_drift_initialization_with_list_of_invalid_image_properties():
    assert_that(
        calling(ImagePropertyDrift).with_args(image_properties=[{'hello': 'string'}]),
        raises(DeepchecksValueError,
               r"Property must be of type dict, and include keys \['name', 'method', 'output_type'\]")
    )


def test_image_property_drift_condition(coco_train_visiondata, coco_test_visiondata, device):
    result = (
        ImagePropertyDrift()
        .add_condition_drift_score_not_greater_than()
        .run(coco_train_visiondata, coco_test_visiondata, device=device)
    )

    assert_that(result, all_of(
        is_correct_image_property_drift_result(),
        contains_passed_condition()
    ))


def test_image_property_drift_fail_condition(coco_train_visiondata, coco_test_visiondata, device):
    result = (
        ImagePropertyDrift()
        .add_condition_drift_score_not_greater_than(0)
        .run(coco_train_visiondata, coco_test_visiondata, device=device)
    )

    assert_that(result, all_of(
        is_correct_image_property_drift_result(),
        contains_failed_condition()
    ))


def contains_failed_condition():
    condition_assertion = has_properties({
        'is_pass': equal_to(False),
        'category': equal_to(ConditionCategory.FAIL),
        'details': matches_regexp(
            r'Earth Mover\'s Distance is above the threshold '
            r'for the next properties\:\n.*'
        )
    })
    return has_property(
        'conditions_results',
        contains_exactly(condition_assertion)
    )


def contains_passed_condition():
    condition_assertion = has_properties({
        'is_pass': equal_to(True),
    })
    return has_property(
        'conditions_results',
        contains_exactly(condition_assertion)
    )


def is_correct_image_property_drift_result():
    value_assertion = all_of(
        instance_of(dict),
        *[has_key(single_property['name']) for single_property in default_image_properties])

    display_assertion = all_of(
        instance_of(list),
        has_length(greater_than(1)),
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


def test_run_on_data_with_only_images(mnist_train_only_images, mnist_test_only_images, device):
    # Act - Assert check runs without exception
    ImagePropertyDrift().run(mnist_train_only_images, mnist_test_only_images, device=device)
