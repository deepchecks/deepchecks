from hamcrest import assert_that, all_of, instance_of, has_key, has_length, has_properties, has_entries, is_, \
    contains_exactly, close_to
from hamcrest.core.matcher import Matcher

from deepchecks import CheckResult
from deepchecks.vision.checks import ImagePropertyOutliers
from deepchecks.vision.utils.image_properties import default_image_properties

from tests.vision.vision_conftest import *


def is_correct_image_property_outliers_result() -> Matcher:
    value_assertion = all_of(
        instance_of(dict),
        *[has_key(single_property['name']) for single_property in default_image_properties])

    display_assertion = all_of(
        instance_of(list),
        has_length(1),
    )

    return all_of(
        instance_of(CheckResult),
        has_properties({
            'value': value_assertion,
            'display': display_assertion
        })
    )


def test_image_property_outliers_check_coco(coco_train_visiondata, device):
    # Run
    result = ImagePropertyOutliers().run(coco_train_visiondata, device=device)

    # Assert
    assert_that(result, is_correct_image_property_outliers_result())
    assert_that(result.value, has_entries({
        'Area': has_entries({
            'sample_values': contains_exactly(139520, 166000, 166500, 187000, 187500, 360000, 361600, 366720, 374544,
                                              378240),
            'count': is_(13),
            'lower_limit': is_(220800),
            'upper_limit': is_(359040)
        }),
        'Mean Red Relative Intensity': instance_of(dict),
        'Mean Green Relative Intensity': instance_of(dict),
        'Mean Blue Relative Intensity': instance_of(dict),
    }))


def test_image_property_outliers_check_mnist(mnist_dataset_train, device):
    # Run
    result = ImagePropertyOutliers().run(mnist_dataset_train, device=device)

    # Assert
    assert_that(result, is_correct_image_property_outliers_result())
    assert_that(result.value, has_entries({
        'Brightness': has_entries({
            'sample_values': has_length(5),
            'count': is_(610),
            'lower_limit': close_to(2.936, .001),
            'upper_limit': close_to(62.650, .001)
        }),
        'Mean Red Relative Intensity': instance_of(str),
        'Mean Green Relative Intensity': instance_of(str),
        'Mean Blue Relative Intensity': instance_of(str),
    }))


def test_run_on_data_with_only_images(mnist_train_only_images, device):
    # Act - Assert check runs without exception
    ImagePropertyOutliers().run(mnist_train_only_images, device=device)
