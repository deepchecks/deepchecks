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
"""Test functions of the VISION model error analysis."""

from hamcrest import assert_that, equal_to, calling, raises

from deepchecks.core.errors import DeepchecksProcessError
from deepchecks.vision.checks import ModelErrorAnalysis
from deepchecks.vision.datasets.classification.mnist import mnist_prediction_formatter
from deepchecks.vision.datasets.detection.coco import yolo_prediction_formatter
from deepchecks.vision.utils import ClassificationPredictionFormatter, DetectionPredictionFormatter


def test_classification(mnist_dataset_train, trained_mnist, device):
    # Arrange
    check = ModelErrorAnalysis(min_error_model_score=0)
    train, test = mnist_dataset_train, mnist_dataset_train

    # Act
    result = check.run(train, test, trained_mnist,
                       prediction_formatter=ClassificationPredictionFormatter(mnist_prediction_formatter),
                       device=device)
    # Assert
    assert_that(len(result.value['feature_segments']), equal_to(1))
    assert_that(result.value['feature_segments']['brightness']['segment1']['n_samples'], equal_to(4498))


def test_detection(coco_train_visiondata, coco_test_visiondata, trained_yolov5_object_detection, device):
    # Arrange
    pred_formatter = DetectionPredictionFormatter(yolo_prediction_formatter)
    check = ModelErrorAnalysis(min_error_model_score=-1)

    # Act
    result = check.run(coco_train_visiondata,
                       coco_test_visiondata,
                       trained_yolov5_object_detection,
                       prediction_formatter=pred_formatter,
                       device=device)
    print(result)

    # Assert
    assert_that(len(result.value['feature_segments']), equal_to(4))
    assert_that(result.value['feature_segments']['normalized_blue_mean']['segment1']['n_samples'], equal_to(48))


def test_classification_not_interesting(mnist_dataset_train, trained_mnist, device):
    # Arrange
    check = ModelErrorAnalysis(min_error_model_score=1)
    train, test = mnist_dataset_train, mnist_dataset_train

    # Assert
    assert_that(calling(check.run).with_args(
        train, test, trained_mnist,
        prediction_formatter=ClassificationPredictionFormatter(mnist_prediction_formatter),
        device=device), raises(DeepchecksProcessError))
