import numpy as np
from mlchecks.checks.performance import ClassificationReport
from hamcrest import *
from mlchecks.utils import MLChecksValueError


def test_dataset_wrong_input():
    X = "wrong_input"
    # Act & Assert
    assert_that(calling(ClassificationReport).with_args(X, None),
                raises(MLChecksValueError, 'dataset must be of type Dataset instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Arrange
    check = ClassificationReport()
    # Assert
    assert_that(calling(ClassificationReport).with_args(iris_dataset, iris_adaboost),
                raises(MLChecksValueError, 'function classification_report requires dataset to have a label column'))


def test_model_info_object(iris_labled_dataset, iris_adaboost):
    # Arrange
    check = ClassificationReport()
    # Act X
    result = check.run(iris_labled_dataset, iris_adaboost).value
    # Assert
    assert(isinstance(len(result.values()) , 3)) # iris has 3 targets
    for col in result.values():
        for val in col.values():
            assert(isinstance(val , float))
