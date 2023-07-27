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

from hamcrest import assert_that, equal_to, greater_than, has_length
from hamcrest import less_than_or_equal_to as le

from deepchecks.vision.checks import ConfusionMatrixReport

# TODO: more tests


def test_classification(mnist_visiondata_train):
    # Arrange
    check = ConfusionMatrixReport(n_samples=None)
    # Act
    result = check.run(mnist_visiondata_train)
    # Assert
    assert_that(result.value.shape, equal_to((10, 10)))
    assert_that(result.display, has_length(greater_than(0)))


def test_classification_without_display(mnist_visiondata_train):
    # Arrange
    check = ConfusionMatrixReport()
    # Act
    result = check.run(mnist_visiondata_train, with_display=False)
    # Assert
    assert_that(result.value.shape, equal_to((10, 10)))
    assert_that(result.display, has_length(0))


def test_classification_not_normalize(mnist_visiondata_train):
    # Arrange
    check = ConfusionMatrixReport(normalized=False)
    # Act
    result = check.run(mnist_visiondata_train)
    # Assert
    assert_that(result.value.shape, equal_to((10, 10)))


def test_detection(coco_visiondata_train):
    # Arrange
    check = ConfusionMatrixReport()
    # Act
    result = check.run(coco_visiondata_train)

    # Assert
    num_of_classes = len(coco_visiondata_train.get_observed_classes()) + 1  # plus no-overlapping
    assert_that(result.value.shape, le((num_of_classes, num_of_classes)))


def test_confusion_matrix_report_display(mnist_visiondata_train):
    # Arrange
    check = ConfusionMatrixReport()

    # Act
    result = check.run(mnist_visiondata_train)

    # Assert
    assert_that(result.display[0], equal_to('Showing 10 of 10 classes:'))
    assert_that(result.display[1],
                equal_to('The overall accuracy of your model is: 97.45%.<br>Best accuracy achieved on samples with <b>'
                         '0</b> label (100.0%).<br>Worst accuracy achieved on samples with <b>9</b> label (86.96%).'))
    # First and second are the text descriptions and third is a heatmap
    assert_that(len(result.display), equal_to(3))
    assert_that(len(result.display[2].data), equal_to(1))
    assert_that(result.display[2].data[0].type, equal_to('heatmap'))
