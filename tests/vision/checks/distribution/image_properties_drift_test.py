from numbers import Number
from hamcrest import (
    assert_that, 
    instance_of, 
    all_of, 
    has_entries, 
    calling,
    raises,
    has_property, 
    has_properties, 
    has_length, 
    contains_exactly,
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

    breakpoint()
    
    assert_that(result, all_of(
        is_correct_image_property_drift_result(),
        contains_passed_condition()
    ))


def test_image_property_drift_condition_with_negative_result():
    train_dataset = coco.load_dataset(train=True, object_type='VisionData')
    
    result = (
        ImagePropertyDrift()
        .add_condition_drift_score_not_greater_than()
        .run(train_dataset, train_dataset)
    )

    breakpoint()

    assert_that(result, all_of(
        is_correct_image_property_drift_result(),
        contains_failed_condition()
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
        instance_of(dict),
        has_entries({
            p: all_of(
                instance_of(dict),
                has_entries({'Drift score': instance_of(Number)})
            )
            for p in ImageFormatter.IMAGE_PROPERTIES
        })
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
    