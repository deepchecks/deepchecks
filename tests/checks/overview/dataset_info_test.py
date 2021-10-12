"""
Contains unit tests for the dataset_info check
"""
from mlchecks.checks.overview.dataset_info import *
from hamcrest import *
from mlchecks.utils import MLChecksValueError


def test_assert_dataset_info(iris_dataset):
    # Act
    result = dataset_info(iris_dataset)
    # Assert
    assert_that(result.value, equal_to((150, 5)))


def test_dataset_wrong_input():
    X = "wrong_input"
    # Act & Assert
    assert_that(calling(dataset_info).with_args(X),
                raises(MLChecksValueError, 'dataset must be of type DataFrame or Dataset. instead got: str'))


def test_dataset_info_object(iris_dataset):
    # Arrange
    di = DatasetInfo()
    # Act
    result = di.run(iris_dataset, model=None)
    # Assert
    assert_that(result.value, equal_to((150, 5)))


def test_dataset_info_dataframe(iris):
    # Act
    result = dataset_info(iris)
    # Assert
    assert_that(result.value, equal_to((150, 5)))