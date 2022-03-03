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
"""Test functions of the VISION train test prediction drift."""
from hamcrest import assert_that, has_entries, close_to, equal_to, raises, calling

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks import TrainTestPredictionDrift
from deepchecks.vision.datasets.classification.mnist import mnist_prediction_formatter
from deepchecks.vision.datasets.detection.coco import yolo_prediction_formatter
from deepchecks.vision.utils import ClassificationPredictionFormatter, DetectionPredictionFormatter


def test_no_drift_classification(mnist_dataset_train, trained_mnist, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_train
    check = TrainTestPredictionDrift()

    # Act
    result = check.run(train, test, trained_mnist,
                       prediction_formatter=ClassificationPredictionFormatter(mnist_prediction_formatter),
                       device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': 0,
             'Method': equal_to('PSI')}
        )}
    ))


def test_no_drift_object_detection(coco_train_visiondata, trained_yolov5_object_detection, device):
    # Arrange
    check = TrainTestPredictionDrift()

    # Act
    result = check.run(coco_train_visiondata, coco_train_visiondata, trained_yolov5_object_detection,
                       prediction_formatter=DetectionPredictionFormatter(yolo_prediction_formatter),
                       device=device)

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


def test_with_drift_classification(mnist_dataset_train, mnist_dataset_test, trained_mnist, device):
    # Arrange
    train, test = mnist_dataset_train, mnist_dataset_test
    check = TrainTestPredictionDrift()

    # Act
    result = check.run(train, test, trained_mnist,
                       prediction_formatter=ClassificationPredictionFormatter(mnist_prediction_formatter),
                       device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': close_to(0, 0.001),
             'Method': equal_to('PSI')}
        )
        }
    ))


def test_with_drift_object_detection(coco_train_visiondata, coco_test_visiondata, trained_yolov5_object_detection,
                                     device):
    # Arrange
    check = TrainTestPredictionDrift()

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata, trained_yolov5_object_detection,
                       prediction_formatter=DetectionPredictionFormatter(yolo_prediction_formatter), device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': close_to(0.31, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding box area (in pixels)': has_entries(
            {'Drift score': close_to(0.009, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of bounding boxes per image': has_entries(
            {'Drift score': close_to(0.054, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_with_drift_object_detection_change_max_cat(coco_train_visiondata, coco_test_visiondata,
                                                    trained_yolov5_object_detection, device):
    # Arrange
    check = TrainTestPredictionDrift(max_num_categories=100)

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata, trained_yolov5_object_detection,
                       prediction_formatter=DetectionPredictionFormatter(yolo_prediction_formatter), device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'Samples per class': has_entries(
            {'Drift score': close_to(0.48, 0.01),
             'Method': equal_to('PSI')}
        ), 'Bounding box area (in pixels)': has_entries(
            {'Drift score': close_to(0.009, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        ), 'Number of bounding boxes per image': has_entries(
            {'Drift score': close_to(0.054, 0.001),
             'Method': equal_to('Earth Mover\'s Distance')}
        )
        }
    ))


def test_with_drift_object_detection_alternative_measurements(coco_train_visiondata, coco_test_visiondata,
                                                              trained_yolov5_object_detection, device):
    # Arrange
    alternative_measurements = [
        {'name': 'test', 'method': lambda x, dataset: int(x[0][0]) if len(x) != 0 else 0, 'is_continuous': True}]
    check = TrainTestPredictionDrift(alternative_prediction_measurements=alternative_measurements)

    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata, trained_yolov5_object_detection,
                       prediction_formatter=DetectionPredictionFormatter(yolo_prediction_formatter), device=device)

    # Assert
    assert_that(result.value, has_entries(
        {'test': has_entries(
            {'Drift score': close_to(0.037, 0.001),
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
    assert_that(calling(TrainTestPredictionDrift).with_args(alternative_measurements),
                raises(DeepchecksValueError,
                       "Measurement must be of type dict, and include keys \['name', 'method', 'is_continuous'\]")
                )


def test_with_drift_object_detection_defected_alternative_measurements2():
    # Arrange
    alternative_measurements = {'name': 'test', 'method': lambda x, dataset: x, 'is_continuous': True}

    # Assert
    assert_that(calling(TrainTestPredictionDrift).with_args(alternative_measurements),
                raises(DeepchecksValueError,
                       "Expected measurements to be a list, instead got dict")
                )
