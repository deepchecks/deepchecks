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
from hamcrest import (
    assert_that,
    instance_of,
    all_of,
    calling,
    raises,
    has_property,
    has_properties,
    has_length,
    contains_exactly,
    greater_than,
    equal_to, has_key, has_entries, close_to
)

from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.utils.image_properties import default_image_properties
from deepchecks.vision.checks.distribution import ImagePropertyDrift
from deepchecks.vision.datasets.detection import coco


def test_image_property_drift_check(device):
    train_dataset = coco.load_dataset(train=True, object_type='VisionData')
    test_dataset = coco.load_dataset(train=False, object_type='VisionData')

    # Run
    result = ImagePropertyDrift().run(train_dataset, test_dataset, device=device)

    # Assert
    assert_that(result, is_correct_image_property_drift_result())

    assert_that(result.value, has_entries(
        {'Brightness': close_to(0.05, 0.01)}
    ))


def test_image_property_drift_check_limit_classes(device):
    train_dataset = coco.load_dataset(train=True, object_type='VisionData')
    test_dataset = coco.load_dataset(train=False, object_type='VisionData')

    # Run
    result = ImagePropertyDrift(classes_to_display=['person', 'cat', 'cell phone', 'car'], min_samples=5
                                ).run(train_dataset, test_dataset, device=device)

    # Assert
    assert_that(result, is_correct_image_property_drift_result())

    assert_that(result.value, has_entries(
        {'Brightness': close_to(0.15, 0.01)}
    ))


def test_image_property_drift_check_limit_classes_illegal(device):
    train_dataset = coco.load_dataset(train=True, object_type='VisionData')
    test_dataset = coco.load_dataset(train=False, object_type='VisionData')
    check = ImagePropertyDrift(classes_to_display=['phone'])
    assert_that(
        calling(check.run).with_args(train_dataset, test_dataset, device=device),
        raises(DeepchecksValueError,  r'Provided list of class ids to display \[\'phone\'\] not found in training '
                                      r'dataset.')
    )


def test_image_property_drift_initialization_with_empty_list_of_image_properties():
    assert_that(
        calling(ImagePropertyDrift).with_args(alternative_image_properties=[]),
        raises(DeepchecksValueError, 'Properties list can\'t be empty')
    )


def test_image_property_drift_initialization_with_list_of_invalid_image_properties():
    assert_that(
        calling(ImagePropertyDrift).with_args(alternative_image_properties=[{'hello': 'string'}]),
        raises(DeepchecksValueError,
               r"Property must be of type dict, and include keys \['name', 'method', 'output_type'\]")
    )


def test_image_property_drift_condition(device):
    train_dataset = coco.load_dataset(train=True, object_type='VisionData')
    test_dataset = coco.load_dataset(train=False, object_type='VisionData')

    result = (
        ImagePropertyDrift()
        .add_condition_drift_score_not_greater_than()
        .run(train_dataset, test_dataset, device=device)
    )

    assert_that(result, all_of(
        is_correct_image_property_drift_result(),
        contains_passed_condition()
    ))


# def contains_failed_condition():
#     condition_assertion = has_properties({
#         'is_pass': equal_to(False),
#         'details': matches(
#             r'Earth Mover\'s Distance is above the threshold '
#             r'for the next properties\:\n.*'
#         )
#     })
#     return has_property(
#         'conditions_results',
#         contains_exactly(condition_assertion)
#     )


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
