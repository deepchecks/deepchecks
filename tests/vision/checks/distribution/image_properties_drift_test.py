from numbers import Number
from hamcrest import (
    assert_that, 
    instance_of, 
    all_of, 
    has_entries, 
    calling,
    raises,
    has_property, 
    has_length, 
    greater_than,
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
        has_property('value', value_assertion),
        has_property('header', 'Image Property Drift'),
        has_property('display', display_assertion)
    )
    