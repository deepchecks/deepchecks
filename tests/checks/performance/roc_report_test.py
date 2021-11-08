"""Contains unit tests for the roc_report check."""
import numpy as np
from mlchecks.checks.performance import RocReport
from mlchecks.utils import MLChecksValueError
from hamcrest import assert_that, calling, raises


def test_dataset_wrong_input():
    bad_dataset = "wrong_input"
    # Act & Assert
    assert_that(calling(RocReport().run).with_args(bad_dataset, None),
                raises(MLChecksValueError,
                       "Check RocReport requires dataset to be of type Dataset. instead got: str"))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(calling(RocReport().run).with_args(iris_dataset, iris_adaboost),
                raises(MLChecksValueError, "Check RocReport requires dataset to have a label column"))



def test_regresion_model(diabetes_split_dataset_and_model):
    # Assert
    train, _, clf = diabetes_split_dataset_and_model
    assert_that(calling(RocReport().run).with_args(train, clf),
                raises(MLChecksValueError, r"Check RocReport Expected model to be a type from"
                                           r" \['multiclass', 'binary'\], but received model of type 'regression'"))


def test_model_info_object(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = RocReport()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert len(result) == 3  # iris has 3 targets
    for value in result.values():
        assert isinstance(value, np.float64)
