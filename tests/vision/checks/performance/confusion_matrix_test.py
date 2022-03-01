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

from hamcrest import assert_that,equal_to

from deepchecks.vision.checks import ConfusionMatrixReport
from deepchecks.vision.datasets.classification.mnist import mnist_prediction_formatter
from deepchecks.vision.utils import ClassificationPredictionFormatter, DetectionPredictionFormatter


def test_classification(mnist_dataset_train, trained_mnist):
    # Arrange
    check = ConfusionMatrixReport()
    # Act
    result = check.run(mnist_dataset_train, trained_mnist,
                       prediction_formatter=ClassificationPredictionFormatter(mnist_prediction_formatter))
    # Assert
    assert_that(result.value.shape, equal_to((10, 10)))


def test_detection(coco_train_visiondata, mock_trained_yolov5_object_detection, simple_prediction_formatter):
    # Arrange
    pred_formatter = DetectionPredictionFormatter(simple_prediction_formatter)
    check = ConfusionMatrixReport()
    # Act
    result = check.run(coco_train_visiondata,
                       mock_trained_yolov5_object_detection,
                       prediction_formatter=pred_formatter)

    # Assert
    assert_that(result.value.shape, equal_to((81, 81)))
