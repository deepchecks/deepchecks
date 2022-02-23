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
import pandas as pd
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
    contains_inanyorder,
    greater_than,
    matches_regexp as matches,
    equal_to
)

from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.utils import ImageFormatter
from deepchecks.vision.checks.distribution import ImagePropertyDrift
from deepchecks.vision.datasets.detection import coco



def test_image_property_drift_check():
    train_dataset = coco.load_dataset(train=True, object_type='VisionData')
    test_dataset = coco.load_dataset(train=False, object_type='VisionData')
    result = ImagePropertyDrift().run(train_dataset, test_dataset)
    assert_that(result, is_correct_image_property_drift_result())


def test_image_property_drift_initialization_with_empty_list_of_image_properties():
    assert_that(
        calling(ImagePropertyDrift).with_args(image_properties=[]),
        raises(DeepchecksValueError, r'image_properties list cannot be empty')
    )


def test_image_property_drift_initialization_with_list_of_unknown_image_properties():
    assert_that(
        calling(ImagePropertyDrift).with_args(image_properties=['hello', 'aspect_ratio']),
        raises(DeepchecksValueError, r'receivedd list of unknown image properties - \[\'hello\'\]')
    )


def test_image_property_drift_condition():
    train_dataset = coco.load_dataset(train=True, object_type='VisionData')
    test_dataset = coco.load_dataset(train=False, object_type='VisionData')

    result = (
        ImagePropertyDrift()
        .add_condition_drift_score_not_greater_than()
        .run(train_dataset, test_dataset)
    )

    assert_that(result, all_of(
        is_correct_image_property_drift_result(),
        contains_passed_condition()
    ))


def contains_failed_condition():
    condition_assertion = has_properties({
        'is_pass': equal_to(False),
        'details': matches(
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
        instance_of(pd.DataFrame),
        has_property('index', contains_inanyorder(*list(ImageFormatter.IMAGE_PROPERTIES))),
    )
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
