"""Contains unit tests for the string_length_out_of_bounds check."""
import numpy as np
import pandas as pd
from mlchecks.base import Dataset

from mlchecks.checks import StringLengthOutOfBounds

from hamcrest import assert_that, has_length


def test_no_outliers():
    # Arrange
    col_data = ['a', 'b'] * 100
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = StringLengthOutOfBounds().run(df).value
    # Assert
    assert_that(result, has_length(0))


def test_single_outlier():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcd'*1000)
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = StringLengthOutOfBounds().run(df).value
    # Assert
    assert_that(result, has_length(1))


def test_outlier_multi_column():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcd'*1000)
    data = {'col1': ['hi']*201,
            'col2': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = StringLengthOutOfBounds().run(df).value
    # Assert
    assert_that(result, has_length(1))


def test_outlier_multiple_outliers():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcd')
    col_data.append('abcd')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = StringLengthOutOfBounds().run(df).value
    # Assert
    assert_that(result, has_length(1))
    assert_that(result['Number of Outlier Samples'][0], 2)


def test_outlier_multiple_outlier_ranges():
    # Arrange
    col_data = ['abcd', 'efgh'] * 100
    col_data.append('a')
    col_data.append('abcdbcdbcdb')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = StringLengthOutOfBounds().run(df).value
    # Assert
    assert_that(result, has_length(2))


def test_fi_n_top(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[0, 'age'] = 'aaa' * 1000
    train.data.loc[0, 'bmi'] = 'aaa' * 1000
    train.data.loc[0, 'bp'] = 'aaa' * 1000
    train.data.loc[0, 'sex'] = 'aaa' * 1000
    # Arrange
    check = StringLengthOutOfBounds(n_top_columns=3)
    # Act
    result_ds = check.run(train, clf).value
    # Assert
    assert_that(result_ds, has_length(3))


def test_nan():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcd')
    col_data.append('abcd')
    col_data.append(np.nan)
    col_data.append(np.nan)
    col_data.append(np.nan)
    col_data.append(np.nan)
    col_data.append(np.nan)
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = StringLengthOutOfBounds().run(df).value
    # Assert
    assert_that(result, has_length(1))
    assert_that(result['Number of Outlier Samples'][0], 2)
