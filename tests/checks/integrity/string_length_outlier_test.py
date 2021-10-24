"""Contains unit tests for the string_mismatch check."""
import pandas as pd

from mlchecks.checks import string_length_outlier

from hamcrest import assert_that, has_length


def test_no_outliers():
    # Arrange
    col_data = ['a', 'b'] * 100
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = string_length_outlier(df).value
    # Assert
    assert_that(result, has_length(0))


def test_single_outlier():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcd'*1000)
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = string_length_outlier(df).value
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
    result = string_length_outlier(df).value
    # Assert
    assert_that(result, has_length(1))


def test_outlier_mutiple_outliers():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcd')
    col_data.append('abcd')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = string_length_outlier(df).value
    # Assert
    assert_that(result, has_length(1))
    assert_that(result['Number of Outlier Samples'][0], 2)

def test_outlier_mutiple_outlier_ranges():
    # Arrange
    col_data = ['abcd', 'efgh'] * 100
    col_data.append('a')
    col_data.append('abcdbcdbcdb')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = string_length_outlier(df).value
    # Assert
    assert_that(result, has_length(2))
