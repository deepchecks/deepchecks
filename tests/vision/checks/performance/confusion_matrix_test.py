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
"""Test functions of the VISION confusion matrix."""

from hamcrest import assert_that, equal_to
from hamcrest import less_than_or_equal_to as le

from deepchecks.vision.checks import ConfusionMatrixReport

# TODO: more tests


def test_classification(mnist_dataset_train, mock_trained_mnist, device):
    # Arrange
    check = ConfusionMatrixReport()
    # Act
    result = check.run(mnist_dataset_train, mock_trained_mnist,
                       device=device)
    # Assert
    assert_that(result.value.shape, equal_to((10, 10)))


def test_detection(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # Arrange
    check = ConfusionMatrixReport()
    # Act
    result = check.run(coco_train_visiondata,
                       mock_trained_yolov5_object_detection,
                       device=device)

    # Assert
    num_of_classes = coco_train_visiondata.num_classes + 1 # plus no-overlapping
    assert_that(result.value.shape, le((num_of_classes, num_of_classes)))
