import numpy as np
from mlchecks.checks.performance import ConfusionMatrixReport
from hamcrest import *
from mlchecks.utils import MLChecksValueError


def test_dataset_wrong_input():
    X = "wrong_input"
    # Act & Assert
    assert_that(calling(ConfusionMatrixReport).with_args(X, None),
                raises(MLChecksValueError, 'dataset must be of type Dataset instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Arrange
    check = ConfusionMatrixReport()
    # Assert
    assert_that(calling(ConfusionMatrixReport).with_args(iris_dataset, iris_adaboost),
                raises(MLChecksValueError, 'function confusion_matrix_report requires dataset to have a label column'))


def test_model_info_object(iris_labled_dataset, iris_adaboost):
    # Arrange
    check = ConfusionMatrixReport()
    # Act X
    result = check.run(iris_labled_dataset, iris_adaboost).value
    # Assert
    for i in range(len(result)):
        for j in range(len(result[i])):
            assert(isinstance(result[i][j] , np.int64))
