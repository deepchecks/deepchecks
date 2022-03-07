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
from hamcrest import assert_that, has_entries, close_to, equal_to, raises, calling
from tests.checks.utils import equal_condition_result

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks import TrainTestLabelDrift


def test_no_drift_classification(mnist_dataset_train, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = TrainTestLabelDrift()

    # Act
    result = check.run(train, test, device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': 0,
             'Method': equal_to('PSI')}
        )}
    ))


def test_no_drift_object_detection(coco_train_visiondata, device):
    # Arrange
    check = TrainTestLabelDrift()

    # Act
    result = check.run(coco_train_visiondata, coco_train_visiondata, device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': 0,
             'Method': equal_to('PSI')}
        ), 'Bounding box area (in pixels)': has_entries(
            {'Drift score': 0,
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of bounding boxes per image': has_entries(
            {'Drift score': 0,
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_with_drift_classification(mnist_dataset_train, mnist_dataset_test, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_test
    check = TrainTestLabelDrift()

    # Act
    result = check.run(train, test, device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': close_to(0, 0.001),
             'Method': equal_to('PSI')}
        )
        }
    ))


def test_with_drift_object_detection(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    check = TrainTestLabelDrift()

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata, device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': close_to(0.24, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding box area (in pixels)': has_entries(
            {'Drift score': close_to(0.012, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of bounding boxes per image': has_entries(
            {'Drift score': close_to(0.059, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_drift_max_drift_score_condition_fail(mnist_drifted_datasets):
    # Arrange
    check = TrainTestLabelDrift().add_condition_drift_score_not_greater_than()
    mod_train_ds, mod_test_ds = mnist_drifted_datasets

    # Act
    result = check.run(mod_train_ds, mod_test_ds)

    condition_result, *_ = result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='PSI <= 0.15 and Earth Mover\'s Distance <= 0.075 for label drift',
        details='Found non-continues label measurements with PSI drift score above threshold: {\'Samples per '
                'class\': \'0.18\'}\n'
    ))


def test_with_drift_object_detection_change_max_cat(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    check = TrainTestLabelDrift(max_num_categories=100)

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata, device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': close_to(0.44, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding box area (in pixels)': has_entries(
            {'Drift score': close_to(0.012, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of bounding boxes per image': has_entries(
            {'Drift score': close_to(0.059, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_with_drift_object_detection_alternative_measurements(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    alternative_measurements = [
        {'name': 'test', 'method': lambda x, dataset: int(x[0][0]) if len(x) != 0 else 0, 'is_continuous': True}]
    check = TrainTestLabelDrift(alternative_label_measurements=alternative_measurements)

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata, device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'test': has_entries(
            {'Drift score': close_to(0.046, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_with_drift_object_detection_defected_alternative_measurements():
    # Arrange
    alternative_measurements = [
        {'name': 'test', 'method': lambda x, dataset: x[0][0] if len(x) != 0 else 0, 'is_continuous': True},
        {'name234': 'test', 'method': lambda x, dataset: x[0][0] if len(x) != 0 else 0, 'is_continuous': True},
    ]

    # Assert
    assert_that(calling(TrainTestLabelDrift).with_args(alternative_measurements),
                raises(DeepchecksValueError,
                       "Measurement must be of type dict, and include keys \['name', 'method', 'is_continuous'\]")
                )


def test_with_drift_object_detection_defected_alternative_measurements2():
    # Arrange
    alternative_measurements = {'name': 'test', 'method': lambda x, dataset: x, 'is_continuous': True}

    # Assert
    assert_that(calling(TrainTestLabelDrift).with_args(alternative_measurements),
                raises(DeepchecksValueError,
                       "Expected measurements to be a list, instead got dict")
                )
