"""Contains unit tests for the dataset_info check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, equal_to, calling, raises

from deepchecks.checks.overview.dataset_info import DatasetInfo
from deepchecks.errors import DeepchecksValueError


def test_assert_dataset_info(iris_dataset):
    # Act
    result = DatasetInfo().run(iris_dataset)
    # Assert
    assert_that(result.value, equal_to((150, 5)))


def test_dataset_wrong_input():
    wrong = 'wrong_input'
    # Act & Assert
    assert_that(calling(DatasetInfo().run).with_args(wrong),
                raises(DeepchecksValueError, 'dataset must be of type DataFrame or Dataset, but got: str'))


def test_dataset_info_object(iris_dataset):
    # Arrange
    di = DatasetInfo()
    # Act
    result = di.run(iris_dataset, model=None)
    # Assert
    assert_that(result.value, equal_to((150, 5)))


def test_dataset_info_dataframe(iris):
    # Act
    result = DatasetInfo().run(iris)
    # Assert
    assert_that(result.value, equal_to((150, 5)))


def test_nan(iris):
    # Act
    df = iris.append(pd.DataFrame({'sepal length (cm)': [np.nan],
                                   'sepal width (cm)':[np.nan],
                                   'petal length (cm)':[np.nan],
                                   'petal width (cm)': [np.nan],
                                   'target':[0]}))
    result = DatasetInfo().run(df)
    # Assert
    assert_that(result.value, equal_to((151, 5)))
