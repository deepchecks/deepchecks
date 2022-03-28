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
from hamcrest import assert_that, all_of, instance_of, has_key, has_length, has_properties, has_entries, is_, \
    contains_exactly, close_to, calling, raises, equal_to
from hamcrest.core.matcher import Matcher

from deepchecks import CheckResult
from deepchecks.core.errors import NotEnoughSamplesError, DeepchecksProcessError
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
    # Act
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
    # Act
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
        'output_type': 'discrete'
    }]
    check = ImagePropertyOutliers(alternative_image_properties=image_properties)
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
        'output_type': 'discrete'
    }]
    check = ImagePropertyOutliers(alternative_image_properties=image_properties)
    # Act - Assert check raise exception
    assert_that(calling(check.run).with_args(mnist_dataset_train, device=device),
                raises(DeepchecksProcessError, 'Image properties are expected to return value per image but instead '
                                               'got 65 values for 64 images for property test'))
