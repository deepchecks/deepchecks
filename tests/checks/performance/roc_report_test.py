import numpy as np
from mlchecks.checks.performance import RocReport
from hamcrest import *
from mlchecks.utils import MLChecksValueError


def test_dataset_wrong_input():
    X = "wrong_input"
    # Act & Assert
    assert_that(calling(RocReport).with_args(X, None),
                raises(MLChecksValueError, 'dataset must be of type Dataset instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Arrange
    check = RocReport()
    # Assert
    assert_that(calling(RocReport).with_args(iris_dataset, iris_adaboost),
                raises(MLChecksValueError, 'function roc_report requires dataset to have a label column'))


def test_model_info_object(iris_labled_dataset, iris_adaboost):
    # Arrange
    check = RocReport()
    # Act X
    result = check.run(iris_labled_dataset, iris_adaboost).value
    # Assert
    assert(isinstance(len(result) , 3)) # iris has 3 targets
    for value in result.values():
        assert(isinstance(value , np.float64))
