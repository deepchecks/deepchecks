"""Contains unit tests for the confusion_matrix_report check."""
import numpy as np
from hamcrest import assert_that, calling, raises
from deepchecks.checks.performance import ConfusionMatrixReport
from deepchecks.errors import DeepchecksValueError


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(ConfusionMatrixReport().run).with_args(bad_dataset, None),
                raises(DeepchecksValueError,
                       'Check ConfusionMatrixReport requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(calling(ConfusionMatrixReport().run).with_args(iris_dataset, iris_adaboost),
                raises(DeepchecksValueError, 'Check ConfusionMatrixReport requires dataset to have a label column'))


def test_regresion_model(diabetes_split_dataset_and_model):
    # Assert
    train, _, clf = diabetes_split_dataset_and_model
    assert_that(calling(ConfusionMatrixReport().run).with_args(train, clf),
                raises(DeepchecksValueError, r'Check ConfusionMatrixReport Expected model to be a type from'
                                           r' \[\'multiclass\', \'binary\'\], but received model of type: regression'))


def test_model_info_object(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = ConfusionMatrixReport()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    for i in range(len(result)):
        for j in range(len(result[i])):
            assert isinstance(result[i][j] , np.int64)
