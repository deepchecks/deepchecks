"""Contains unit tests for the string_length_out_of_bounds check."""
import numpy as np
import pandas as pd
from deepchecks.base import Dataset

from deepchecks.checks import StringLengthOutOfBounds

from hamcrest import assert_that, has_length, has_items

from tests.checks.utils import equal_condition_result


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
    assert_that(result['col1']['outliers'][0]['n_samples'], 2)


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
    assert_that(result, has_length(1))
    assert_that(result['col1']['outliers'], has_length(2))


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
    result_ds = check.run(train, clf).display[0]
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
    assert_that(result['col1']['outliers'][0]['n_samples'], 2)


def test_condition_count_fail():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcd')
    col_data.append('abcd')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    check = StringLengthOutOfBounds().add_condition_number_of_outliers_not_greater_than(1)

    # Act
    result = check.conditions_decision(check.run(df))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found columns with greater than 1 outliers: col1',
                               name='Number of outliers not greater than 1 string length outliers for all columns')
    ))


def test_condition_count_pass():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcd')
    col_data.append('abcd')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    check = StringLengthOutOfBounds().add_condition_number_of_outliers_not_greater_than(10)

    # Act
    result = check.conditions_decision(check.run(df))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Number of outliers not greater than 10 string length outliers for all columns')
    ))


def test_condition_ratio_fail():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcd')
    col_data.append('abcd')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    check = StringLengthOutOfBounds().add_condition_ratio_of_outliers_not_greater_than(0.001)

    # Act
    result = check.conditions_decision(check.run(df))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found columns with greater than 0.10% outliers: col1',
                               name='Ratio of outliers not greater than 0.10% string length outliers for all columns')
    ))


def test_condition_ratio_pass():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcd')
    col_data.append('abcd')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    check = StringLengthOutOfBounds().add_condition_ratio_of_outliers_not_greater_than(0.1)

    # Act
    result = check.conditions_decision(check.run(df))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Ratio of outliers not greater than 10.00% string length outliers for all columns')
    ))
