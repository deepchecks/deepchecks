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
from hamcrest import assert_that, close_to, greater_than, has_length

from deepchecks.vision.checks import HeatmapComparison


def test_object_detection(coco_visiondata_train, coco_visiondata_test):
    # Arrange
    check = HeatmapComparison()

    # Act
    result = check.run(coco_visiondata_train, coco_visiondata_test)

    # Assert
    brightness_diff = result.value["diff"]
    assert_that(brightness_diff.mean(), close_to(10.461, 0.001))
    assert_that(brightness_diff.max(), close_to(44, 0.001))

    bbox_diff = result.value["diff_bbox"]
    assert_that(bbox_diff.mean(), close_to(5.593, 0.001))
    assert_that(bbox_diff.max(), close_to(23, 0.001))


def test_classification(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = HeatmapComparison(n_samples=None)

    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test)

    # Assert
    brightness_diff = result.value["diff"]
    assert_that(brightness_diff.mean(), close_to(6.834, 0.001))
    assert_that(brightness_diff.max(), close_to(42, 0.001))
    assert_that(result.display, has_length(greater_than(0)))


def test_classification_without_display(mnist_visiondata_train, mnist_visiondata_test):
    # Arrange
    check = HeatmapComparison()

    # Act
    result = check.run(mnist_visiondata_train, mnist_visiondata_test, with_display=False)

    # Assert
    brightness_diff = result.value["diff"]
    assert_that(brightness_diff.mean(), close_to(6.834, 0.001))
    assert_that(brightness_diff.max(), close_to(42, 0.001))
    assert_that(result.display, has_length(0))


# def test_classification_limit_classes(mnist_visiondata_train, mnist_visiondata_test):
#     # Arrange
#     check = HeatmapComparison(classes_to_display=['9'])
#
#     # Act
#     result = check.run(mnist_visiondata_train, mnist_visiondata_test)
#
#     # Assert
#     brightness_diff = result.value["diff"]
#     assert_that(brightness_diff.mean(), close_to(12.632, 0.001))
#     assert_that(brightness_diff.max(), close_to(113, 0.001))
#
#
# def test_object_detection_limit_classes(coco_visiondata_train, coco_visiondata_test):
#     # Arrange
#     check = HeatmapComparison(classes_to_display=['bicycle', 'bench'])
#
#     # Act
#     result = check.run(coco_visiondata_train, coco_visiondata_test)
#
#     # Assert
#     brightness_diff = result.value["diff"]
#     assert_that(brightness_diff.mean(), close_to(39.364, 0.001))
#     assert_that(brightness_diff.max(), close_to(164, 0.001))
#
#     bbox_diff = result.value["diff_bbox"]
#     assert_that(bbox_diff.mean(), close_to(15.154, 0.001))
#     assert_that(bbox_diff.max(), close_to(75, 0.001))
#
#
# def test_limit_classes_nonexistant_class(coco_visiondata_train, coco_visiondata_test):
#     # Arrange
#     check = HeatmapComparison(classes_to_display=['1000'])
#
#     # Act & Assert
#     assert_that(
#         calling(check.run).with_args(coco_visiondata_train, coco_visiondata_test),
#         raises(DeepchecksValueError,
#                r'Train dataset does not contain the following classes selected for display: \[\'1000\'\]')
#     )


def test_custom_task(mnist_train_custom_task, mnist_test_custom_task):
    # Arrange
    check = HeatmapComparison(n_samples=None)

    # Act
    result = check.run(mnist_train_custom_task, mnist_test_custom_task)

    # Assert
    brightness_diff = result.value["diff"]
    assert_that(brightness_diff.mean(), close_to(1.095, 0.001))
    assert_that(brightness_diff.max(), close_to(9, 0.001))


def test_dataset_name(mnist_visiondata_train, mnist_visiondata_test):
    mnist_visiondata_train.name = 'Ref'
    mnist_visiondata_test.name = 'Win'

    result = HeatmapComparison().run(mnist_visiondata_train, mnist_visiondata_test)

    assert_that(result.display[0].layout.annotations[0].text, 'Ref')
    assert_that(result.display[0].layout.annotations[1].text, 'Win')
