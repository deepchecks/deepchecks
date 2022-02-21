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
"""Test functions of the heatmap comparison check."""
from hamcrest import assert_that, raises, calling, less_than

from deepchecks.core.errors import DeepchecksValueError, DeepchecksNotSupportedError
from deepchecks.vision.checks.distribution import HeatmapComparison


def test_object_detection(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    check = HeatmapComparison()

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata)

    # Assert
    brightness_diff = result.value["diff"]
    assert_that(brightness_diff.mean(), less_than(11))
    assert_that(brightness_diff.max(), less_than(45))

    bbox_diff = result.value["diff_bbox"]
    assert_that(bbox_diff.mean(), less_than(11))
    assert_that(bbox_diff.max(), less_than(24))


def test_classification(mnist_dataset_train, mnist_dataset_test):
    # Arrange
    check = HeatmapComparison()

    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test)

    # Assert
    brightness_diff = result.value["diff"]
    assert_that(brightness_diff.mean(), less_than(2))
    assert_that(brightness_diff.max(), less_than(10))


def test_object_detection_limit_classes(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    check = HeatmapComparison(classes_to_display=[0])

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata)

    # Assert
    brightness_diff = result.value["diff"]
    assert_that(brightness_diff.mean(), less_than(11))
    assert_that(brightness_diff.max(), less_than(45))

    bbox_diff = result.value["diff_bbox"]
    assert_that(bbox_diff.mean(), less_than(6))
    assert_that(bbox_diff.max(), less_than(22))


def test_limit_classes_for_classification(mnist_dataset_train, mnist_dataset_test):
    # Arrange
    check = HeatmapComparison(classes_to_display=[0])

    # Act & Assert
    assert_that(
        calling(check.run).with_args(mnist_dataset_test, mnist_dataset_test),
        raises(DeepchecksNotSupportedError, 'Classes to display is only supported for object detection tasks.')
    )


def test_limit_classes_nonexistant_class(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    check = HeatmapComparison(classes_to_display=[1000])

    # Act & Assert
    assert_that(
        calling(check.run).with_args(coco_train_visiondata, coco_test_visiondata),
        raises(DeepchecksValueError,
               r'Provided list of class ids to display \[1000\] not found in training dataset.')
    )
