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
from hamcrest import (all_of, any_of, assert_that, calling, close_to, contains_exactly, equal_to, greater_than,
                      has_entries, has_key, has_length, has_properties, instance_of, is_, raises)
from hamcrest.core.matcher import Matcher

from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksProcessError
from deepchecks.vision.checks import ImagePropertyOutliers
from deepchecks.vision.utils.image_properties import default_image_properties
from tests.vision.vision_conftest import *


def is_correct_image_property_outliers_result(with_display: bool = True) -> Matcher:
    value_assertion = all_of(
        instance_of(dict),
        *[has_key(single_property['name']) for single_property in default_image_properties])

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


def test_image_property_outliers_check_coco(coco_train_visiondata, device):
    # Act
    result = ImagePropertyOutliers().run(coco_train_visiondata, device=device)

    # Assert
    assert_that(result, is_correct_image_property_outliers_result())
    assert_that(result.value, has_entries({
        'Area': has_entries({
            'outliers_identifiers': has_length(13),
            'lower_limit': is_(220800),
            'upper_limit': is_(359040)
        }),
        'Mean Red Relative Intensity': instance_of(dict),
        'Mean Green Relative Intensity': instance_of(dict),
        'Mean Blue Relative Intensity': instance_of(dict),
    }))


def test_image_property_outliers_check_coco_without_display(coco_train_visiondata, device):
    # Act
    result = ImagePropertyOutliers().run(coco_train_visiondata, device=device, with_display=False)

    # Assert
    assert_that(result, is_correct_image_property_outliers_result(with_display=False))
    assert_that(result.value, has_entries({
        'Area': has_entries({
            'outliers_identifiers': has_length(13),
            'lower_limit': is_(220800),
            'upper_limit': is_(359040)
        }),
        'Mean Red Relative Intensity': instance_of(dict),
        'Mean Green Relative Intensity': instance_of(dict),
        'Mean Blue Relative Intensity': instance_of(dict),
    }))


def test_image_property_outliers_check_mnist(mnist_dataset_train, device):
    # Act
    result = ImagePropertyOutliers().run(mnist_dataset_train, device=device, n_samples=None)

    # Assert
    assert_that(result, is_correct_image_property_outliers_result())
    assert_that(result.value, has_entries({
        'Brightness': has_entries({
            'outliers_identifiers': has_length(609),
            'lower_limit': close_to(6.45, .01),
            'upper_limit': close_to(62.37, .01)
        }),
        'Mean Red Relative Intensity': instance_of(str),
        'Mean Green Relative Intensity': instance_of(str),
        'Mean Blue Relative Intensity': instance_of(str),
    }))


def test_run_on_data_with_only_images(mnist_train_only_images, device):
    # Act - Assert check runs without exception
    result = ImagePropertyOutliers().run(mnist_train_only_images, device=device, n_samples=None)
    # Assert
    assert_that(result, is_correct_image_property_outliers_result())


def test_run_on_custom_task(mnist_train_custom_task, device):
    # Act - Assert check runs without exception
    result = ImagePropertyOutliers().run(mnist_train_custom_task, device=device, n_samples=None)
    # Assert
    assert_that(result, is_correct_image_property_outliers_result())


def test_not_enough_samples_for_iqr(mnist_dataset_train, device):
    # Arrange
    five_samples_mnist = mnist_dataset_train.copy(n_samples=5, random_state=0)
    check = ImagePropertyOutliers()
    # Act
    result = check.run(five_samples_mnist, device=device)
    # Assert
    assert_that(result, is_correct_image_property_outliers_result())
    assert_that(result.value, has_entries({
        'Brightness': equal_to('Not enough non-null samples to calculate outliers.')
    }))


def test_string_property_exception(mnist_dataset_train, device):
    # Arrange
    def string_property(images):
        return ['test'] * len(images)
    image_properties = [{
        'name': 'test',
        'method': string_property,
        'output_type': 'categorical'
    }]
    check = ImagePropertyOutliers(image_properties=image_properties)
    # Act - Assert check raise exception
    assert_that(calling(check.run).with_args(mnist_dataset_train, device=device),
                raises(DeepchecksProcessError, 'For outliers, properties are expected to be only numeric types but '
                                               'found non-numeric value for property test'))


def test_incorrect_properties_count_exception(mnist_dataset_train, device):
    # Arrange
    def too_many_property(images):
        return ['test'] * (len(images) + 1)
    image_properties = [{
        'name': 'test',
        'method': too_many_property,
        'output_type': 'categorical'
    }]
    check = ImagePropertyOutliers(image_properties=image_properties)
    # Act - Assert check raise exception
    assert_that(calling(check.run).with_args(mnist_dataset_train, device=device),
                raises(DeepchecksProcessError, 'Properties are expected to return value per image but instead got 65 '
                                               'values for 64 images for property test'))


def test_property_with_nones(mnist_dataset_train, device):
    # Arrange
    def property_with_none(images):
        return [[1, None]] * (len(images))
    image_properties = [{
        'name': 'test',
        'method': property_with_none,
        'output_type': 'categorical'
    }]
    check = ImagePropertyOutliers(image_properties=image_properties)
    # Act
    result = check.run(mnist_dataset_train, device=device)
    assert_that(result.value, has_entries({
        'test': has_entries({
            'outliers_identifiers': has_length(0),
            'lower_limit': is_(1),
            'upper_limit': is_(1)
        })
    }))
