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
"""Test functions of the VISION train test prediction drift."""

from hamcrest import assert_that, close_to, equal_to, greater_than, has_entries, has_length

from deepchecks.vision.checks import TrainTestPredictionDrift
from tests.base.utils import equal_condition_result


def test_no_drift_classification(mnist_visiondata_train):
    # Arrange
    check = TrainTestPredictionDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_train)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': 0,
             'Method': equal_to('PSI')}
        )}
    ))


def test_no_drift_object_detection(coco_visiondata_train):
    # Arrange
    check = TrainTestPredictionDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_train)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': 0,
             'Method': equal_to('PSI')}
        ), 'Bounding Box Area (in pixels)': has_entries(
            {'Drift score': 0,
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Bounding Boxes Per Image': has_entries(
            {'Drift score': 0,
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))
    assert_that(result.display, has_length(greater_than(0)))


def test_no_drift_object_detection_without_display(coco_visiondata_train):
    # Arrange
    check = TrainTestPredictionDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_train, with_display=False)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': 0,
             'Method': equal_to('PSI')}
        ), 'Bounding Box Area (in pixels)': has_entries(
            {'Drift score': 0,
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Bounding Boxes Per Image': has_entries(
            {'Drift score': 0,
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))
    assert_that(result.display, has_length(0))


def test_with_drift_classification(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = TrainTestPredictionDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0.057, 0.01),
             'Method': equal_to('PSI')}
        )
        }
    ))


def test_with_drift_segmentation(segmentation_coco_visiondata_train, segmentation_coco_visiondata_test):
    # Arrange
    check = TrainTestPredictionDrift()

    # Act
    result = check.run(segmentation_coco_visiondata_train, segmentation_coco_visiondata_test)
    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': equal_to(0),
             'Method': equal_to('Cramer\'s V')}
        ), 'Number of Classes Per Image': has_entries(
            {'Drift score': close_to(0.1, 0.01),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Segment Area (in pixels)': has_entries(
            {'Drift score': close_to(0.03, 0.01),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_reduce_with_drift_classification(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = TrainTestPredictionDrift(categorical_drift_method='PSI')
    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)
    # Assert
    assert_that(result.reduce_output(), has_entries(
        {'Samples Per Class': close_to(0.057, 0.01)}
    ))


def test_with_drift_classification_cramer(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    train, test = mnist_visiondata_train, mnist_visiondata_test
    check = TrainTestPredictionDrift(categorical_drift_method='cramer_v')

    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0, 0.001),
             'Method': equal_to('Cramer\'s V')}
        )
        }
    ))


def test_with_drift_object_detection(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    check = TrainTestPredictionDrift(categorical_drift_method='PSI', max_num_categories_for_drift=10,
                                     min_category_size_ratio=0)

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_test)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0.37, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding Box Area (in pixels)': has_entries(
            {'Drift score': close_to(0.012, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Bounding Boxes Per Image': has_entries(
            {'Drift score': close_to(0.085, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_with_drift_object_detection_change_max_cat(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    check = TrainTestPredictionDrift(categorical_drift_method='PSI', max_num_categories_for_drift=100,
                                     min_category_size_ratio=0)

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_test)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0.48, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding Box Area (in pixels)': has_entries(
            {'Drift score': close_to(0.012, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Bounding Boxes Per Image': has_entries(
            {'Drift score': close_to(0.085, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_with_drift_object_detection_alternative_measurements(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    def prop(predictions):
        return [int(x[0][0]) if len(x) != 0 else 0 for x in predictions]

    alternative_measurements = [
        {'name': 'test', 'method': prop, 'output_type': 'numerical'}]
    check = TrainTestPredictionDrift(categorical_drift_method='PSI', prediction_properties=alternative_measurements)

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_test)

    # Assert
    assert_that(result.value, has_entries(
        {'test': has_entries(
            {'Drift score': close_to(0.046, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_drift_max_drift_score_condition_fail(mnist_drifted_datasets):
    # Arrange
    check = TrainTestPredictionDrift(categorical_drift_method='PSI').add_condition_drift_score_less_than()
    mod_train_ds, mod_test_ds = mnist_drifted_datasets

    # Act
    result = check.run(mod_train_ds, mod_test_ds, random_state=42)

    condition_result, *_ = result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='categorical drift score < 0.15 and numerical drift score < 0.075 for prediction drift',
        details='Failed for 1 out of 1 prediction properties.\n'
                'Found 1 categorical prediction properties with PSI above threshold: {\'Samples Per Class\': \'0.39\'}'
    ))


def test_condition_pass(mnist_visiondata_train):
    # Arrange
    check = TrainTestPredictionDrift(categorical_drift_method='PSI').add_condition_drift_score_less_than()

    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_train)

    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        name='categorical drift score < 0.15 and numerical drift score < 0.075 for prediction drift',
        details='Passed for 1 prediction properties out of 1 prediction properties.\n'
                'Found prediction property "Samples Per Class" has the highest categorical drift score: 0'
    ))
