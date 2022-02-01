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


def test_no_drift_classification_label(mnist_dataset_train, mnist_dataset_test):
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


def test_no_drift_object_detection_label(coco_dataset):
    # Arrange
    train, test = coco_dataset, coco_dataset
    check = TrainTestLabelDrift()

    # Act
    result = check.run(train, test)

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
             'Method': equal_to('PSI')}
        )
        }
    )
                )
