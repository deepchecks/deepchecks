import torch
from ignite.metrics import Precision, Accuracy
from hamcrest import (assert_that, equal_to, greater_than_or_equal_to,
                      is_in, raises, calling)

from deepchecks.vision.checks.performance.single_dataset_scalar_performance import SingleDatasetScalarPerformance


def test_detection_deafults(coco_train_visiondata, mock_trained_yolov5_object_detection):
    # Arrange
    check = SingleDatasetScalarPerformance()

    # Act
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection)

    # Assert
    # scalar
    assert_that(type(result.value), equal_to(float))
    # metric
    assert_that(result.value, greater_than_or_equal_to(0))


def test_classification_defaults(mnist_dataset_train, mock_trained_mnist):
    # Arrange
    check = SingleDatasetScalarPerformance()

    # Act
    result = check.run(mnist_dataset_train, mock_trained_mnist)

    # Assert
    # scalar
    assert_that(type(result.value), equal_to(float))
    # metric
    assert_that(result.value, greater_than_or_equal_to(0))


def test_add_condition(mnist_dataset_train, mock_trained_mnist):
    # Arrange
    check = SingleDatasetScalarPerformance()
    check.add_condition_greater_than(0.5)
    check.add_condition_less_equal_to(0.2)

    # Act
    result = check.run(mnist_dataset_train, mock_trained_mnist)

    # Assert
    assert_that(result.conditions_results[0].is_pass)
    assert_that(result.conditions_results[1].is_pass, equal_to(False))


def test_classification_w_params(mnist_dataset_train, mock_trained_mnist):
    check = SingleDatasetScalarPerformance(Precision(), torch.max)
    result = check.run(mnist_dataset_train, mock_trained_mnist)
    assert_that(type(result.value), equal_to(float))


