"""Contains unit tests for the calibration_metric check."""
import numpy as np
from mlchecks.checks.performance import CalibrationMetric, calibration_metric
from mlchecks.utils import MLChecksValueError
from hamcrest import assert_that, calling, raises


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(calibration_metric).with_args(bad_dataset, None),
                raises(MLChecksValueError,
                       'function calibration_metric requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(calling(calibration_metric).with_args(iris_dataset, iris_adaboost),
                raises(MLChecksValueError, 'function calibration_metric requires dataset to have a label column'))


def test_model_info_object(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = CalibrationMetric()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert len(result) == 3 # iris has 3 targets
    for value in result.values():
        assert isinstance(value , np.float64)
