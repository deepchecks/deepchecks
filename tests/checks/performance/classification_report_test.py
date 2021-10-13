import numpy as np
from mlchecks.checks.performance import ClassificationReport, classification_report
from hamcrest import *
from mlchecks.utils import MLChecksValueError


def test_dataset_wrong_input():
    X = "wrong_input"
    # Act & Assert
    assert_that(calling(classification_report).with_args(X, None),
                raises(MLChecksValueError, 'function classification_report requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(calling(classification_report).with_args(iris_dataset, iris_adaboost),
                raises(MLChecksValueError, 'function classification_report requires dataset to have a label column'))


def test_model_info_object(iris_labled_dataset, iris_adaboost):
    # Arrange
    check = ClassificationReport()
    # Act X
    result = check.run(iris_labled_dataset, iris_adaboost).value
    # Assert
    assert(len(result.values()) == 3) # iris has 3 targets
    for col in result.values():
        for val in col.values():
            assert(isinstance(val , float))
