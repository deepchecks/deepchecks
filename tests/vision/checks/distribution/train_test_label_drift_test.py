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
from hamcrest import assert_that, has_entries, close_to, equal_to

from deepchecks.vision.checks import TrainTestLabelDrift
from tests.vision.vision_conftest import *


def test_no_drift_classification(mnist_dataset_train):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = TrainTestLabelDrift()

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': 0,
             'Method': equal_to('PSI')}
        )
        }
    )
                )


def test_no_drift_object_detection(coco_train_visiondata):
    # Arrange
    check = TrainTestLabelDrift()

    # Act
    result = check.run(coco_train_visiondata, coco_train_visiondata)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': 0,
             'Method': equal_to('PSI')}
        ), 'Bounding box area distribution': has_entries(
            {'Drift score': 0,
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of bounding boxes per image': has_entries(
            {'Drift score': 0,
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    )
                )


def test_with_drift_classification(mnist_dataset_train, mnist_dataset_test):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_test
    check = TrainTestLabelDrift()

    # Act
    result = check.run(train, test)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': close_to(0, 0.001),
             'Method': equal_to('PSI')}
        )
        }
    )
                )


def test_with_drift_object_detection(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    check = TrainTestLabelDrift()

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': close_to(0.44, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding box area distribution': has_entries(
            {'Drift score': close_to(0.012, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of bounding boxes per image': has_entries(
            {'Drift score': close_to(0.058, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    )
                )


def test_with_drift_object_detection_changed_num_bins(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    check = TrainTestLabelDrift(num_bins=10)

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': close_to(0.44, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding box area distribution': has_entries(
            {'Drift score': close_to(0.01, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of bounding boxes per image': has_entries(
            {'Drift score': close_to(0.043, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    )
                )


def test_with_drift_object_detection_alternative_measurements(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    alternative_measurements = [
        {'name': 'test', 'method': lambda x: x[0][0] if len(x) != 0 else 0, 'is_continuous': True}]
    check = TrainTestLabelDrift(alternative_label_measurements=alternative_measurements)

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata)

    # Assert
    assert_that(result.value, has_entries(
        {'test': has_entries(
            {'Drift score': close_to(0.046, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    )
                )
