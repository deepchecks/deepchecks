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
"""Test functions of the VISION train test label drift."""
from hamcrest import assert_that, close_to, equal_to, greater_than, has_entries, has_length

from deepchecks.vision.checks import TrainTestLabelDrift
from tests.base.utils import equal_condition_result


def test_no_drift_classification(mnist_visiondata_train):
    # Arrange
    check = TrainTestLabelDrift(categorical_drift_method='PSI')

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
    check = TrainTestLabelDrift(categorical_drift_method='PSI')

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


def test_no_drift_object_segmentation(segmentation_coco_visiondata_train):
    # Arrange
    check = TrainTestLabelDrift()

    # Act
    result = check.run(segmentation_coco_visiondata_train, segmentation_coco_visiondata_train)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': 0,
             'Method': equal_to('Cramer\'s V')}
        ), 'Segment Area (in pixels)': has_entries(
            {'Drift score': 0,
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Classes Per Image': has_entries(
            {'Drift score': 0,
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_reduce_output_no_drift_object_detection(coco_visiondata_train):
    # Arrange
    check = TrainTestLabelDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_train)

    # Assert
    assert_that(result.reduce_output(), has_entries(
        {'Samples Per Class': 0,
         'Bounding Box Area (in pixels)': 0, 'Number of Bounding Boxes Per Image': 0}
    ))


def test_with_drift_classification(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    train, test = mnist_visiondata_train, mnist_visiondata_test
    check = TrainTestLabelDrift(categorical_drift_method='PSI')

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0.054, 0.001),
             'Method': equal_to('PSI')}
        )
        }
    ))


def test_with_drift_classification_cramer(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    train, test = mnist_visiondata_train, mnist_visiondata_test
    check = TrainTestLabelDrift(categorical_drift_method='cramer_v')

    # Act
    result = check.run(train, test)

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
    check = TrainTestLabelDrift(categorical_drift_method='PSI', max_num_categories_for_drift=10,
                                min_category_size_ratio=0)

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_test)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0.37, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding Box Area (in pixels)': has_entries(
            {'Drift score': close_to(0.013, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Bounding Boxes Per Image': has_entries(
            {'Drift score': close_to(0.051, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))
    assert_that(result.display, has_length(greater_than(0)))


def test_with_drift_object_detection_without_display(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    check = TrainTestLabelDrift(categorical_drift_method='PSI', max_num_categories_for_drift=10,
                                min_category_size_ratio=0)

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_test, with_display=False)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0.37, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding Box Area (in pixels)': has_entries(
            {'Drift score': close_to(0.013, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Bounding Boxes Per Image': has_entries(
            {'Drift score': close_to(0.051, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))
    assert_that(result.display, has_length(0))


def test_drift_max_drift_score_condition_fail(mnist_drifted_datasets):
    # Arrange
    check = TrainTestLabelDrift(categorical_drift_method='PSI') \
        .add_condition_drift_score_less_than(max_allowed_categorical_score=0.1)
    mod_train_ds, mod_test_ds = mnist_drifted_datasets

    # Act
    result = check.run(mod_train_ds, mod_test_ds)
    condition_result, *_ = result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='categorical drift score < 0.1 and numerical drift score < 0.075',
        details='Failed for 1 out of 1 label properties.\nFound 1 categorical label properties with PSI above '
                'threshold: {\'Samples Per Class\': \'0.32\'}'
    ))


def test_drift_max_drift_score_condition_fail_cremer_v(mnist_drifted_datasets):
    # Arrange
    check = TrainTestLabelDrift(categorical_drift_method='cramer_v') \
        .add_condition_drift_score_less_than(max_allowed_categorical_score=0.1)
    mod_train_ds, mod_test_ds = mnist_drifted_datasets

    # Act
    result = check.run(mod_train_ds, mod_test_ds)

    condition_result, *_ = result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='categorical drift score < 0.1 and numerical drift score < 0.075',
        details='Failed for 1 out of 1 label properties.\n'
                'Found 1 categorical label properties with Cramer\'s V above threshold: {\'Samples Per '
                'Class\': \'0.22\'}'
    ))


def test_with_drift_object_detection_change_max_cat(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    check = TrainTestLabelDrift(categorical_drift_method='PSI', max_num_categories_for_drift=100)

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_test)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0.44, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding Box Area (in pixels)': has_entries(
            {'Drift score': close_to(0.013, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Bounding Boxes Per Image': has_entries(
            {'Drift score': close_to(0.051, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_display_changes_but_values_dont_for_diff_display_params(coco_visiondata_train, coco_visiondata_test):
    # Arrange and assert
    check = TrainTestLabelDrift(categorical_drift_method='PSI', min_category_size_ratio=0,
                                max_num_categories_for_display=20, show_categories_by='test_largest')
    result = check.run(coco_visiondata_train, coco_visiondata_test)
    assert_that(result.value, has_entries(
        {'Samples Per Class': has_entries(
            {'Drift score': close_to(0.44, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding Box Area (in pixels)': has_entries(
            {'Drift score': close_to(0.013, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of Bounding Boxes Per Image': has_entries(
            {'Drift score': close_to(0.051, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }))


def test_with_drift_object_detection_alternative_properties(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    def prop(labels):
        return [int(x[0][0]) if len(x) != 0 else 0 for x in labels]

    alternative_properties = [
        {'name': 'test', 'method': prop, 'output_type': 'numerical'}]
    check = TrainTestLabelDrift(label_properties=alternative_properties)

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_test)

    # Assert
    assert_that(result.value, has_entries(
        {'test': has_entries(
            {'Drift score': close_to(0.05, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))
