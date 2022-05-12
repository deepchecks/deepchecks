import torch
from ignite.metrics import Precision
from hamcrest import (assert_that, equal_to, greater_than_or_equal_to,
                      is_in, raises)

from deepchecks.vision.checks.performance.single_dataset_scalar_performance import SingleDatasetScalarPerformance


def test_classification(mnist_dataset_train, mock_trained_mnist, device):
    # Arrange
    check = SingleDatasetScalarPerformance(Precision())

    # Act
    result = check.run(mnist_dataset_train, mock_trained_mnist)

    # Assert
    # scalar
    assert_that(result.value.dim(), equal_to(0))
    # metric
    assert_that(result.value, greater_than_or_equal_to(0))