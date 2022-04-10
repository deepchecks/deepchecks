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
from hamcrest import assert_that, raises, calling, close_to

from deepchecks.core.errors import DeepchecksValueError, DeepchecksNotSupportedError
from deepchecks.vision.checks.distribution import HeatmapComparison


def test_object_detection(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    check = HeatmapComparison()

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata, device=device)

    # Assert
    brightness_diff = result.value["diff"]
    assert_that(brightness_diff.mean(), close_to(10.420, 0.001))
    assert_that(brightness_diff.max(), close_to(45, 0.001))

    bbox_diff = result.value["diff_bbox"]
    assert_that(bbox_diff.mean(), close_to(5.589, 0.001))
    assert_that(bbox_diff.max(), close_to(24, 0.001))


def test_classification(mnist_dataset_train, mnist_dataset_test, device):
    # Arrange
    check = HeatmapComparison()

    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, device=device, n_samples=None)

    # Assert
    brightness_diff = result.value["diff"]
    assert_that(brightness_diff.mean(), close_to(1.095, 0.001))
    assert_that(brightness_diff.max(), close_to(9, 0.001))


def test_classification_limit_classes(mnist_dataset_train, mnist_dataset_test, device):
    # Arrange
    check = HeatmapComparison(classes_to_display=['9'])

    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, device=device, n_samples=None)

    # Assert
    brightness_diff = result.value["diff"]
    assert_that(brightness_diff.mean(), close_to(2.149, 0.001))
    assert_that(brightness_diff.max(), close_to(21, 0.001))


def test_object_detection_limit_classes(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    check = HeatmapComparison(classes_to_display=['person', 'cat'])

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata, device=device)

    # Assert
    brightness_diff = result.value["diff"]
    assert_that(brightness_diff.mean(), close_to(49, 0.001))
    assert_that(brightness_diff.max(), close_to(208, 0.001))

    bbox_diff = result.value["diff_bbox"]
    assert_that(bbox_diff.mean(), close_to(18.501, 0.001))
    assert_that(bbox_diff.max(), close_to(100, 0.001))


def test_limit_classes_nonexistant_class(coco_train_visiondata, coco_test_visiondata, device):
    # Arrange
    check = HeatmapComparison(classes_to_display=['1000'])

    # Act & Assert
    assert_that(
        calling(check.run).with_args(coco_train_visiondata, coco_test_visiondata, device=device),
        raises(DeepchecksValueError,
               r'Provided list of class ids to display \[\'1000\'\] not found in training dataset.')
    )


def test_custom_task(mnist_train_custom_task, mnist_test_custom_task, device):
    # Arrange
    check = HeatmapComparison()

    # Act
    result = check.run(mnist_train_custom_task, mnist_test_custom_task, device=device, n_samples=None)

    # Assert
    brightness_diff = result.value["diff"]
    assert_that(brightness_diff.mean(), close_to(1.095, 0.001))
    assert_that(brightness_diff.max(), close_to(9, 0.001))
