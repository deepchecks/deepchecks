"""Tests for Single Value Check"""
import pandas as pd
from mlchecks.utils import MLChecksValueError
from mlchecks.checks.integrity.is_single_value import IsSingleValue
from hamcrest import assert_that, calling, raises, equal_to


def helper_test_df_and_result(df, expected_result_value, ignore_columns=None):
    # Act
    result = IsSingleValue(ignore_columns=ignore_columns).run(df)

    # Assert
    assert_that(result.value, equal_to(expected_result_value))


def test_single_column_dataset_more_than_single_value():
    # Arrange
    df = pd.DataFrame({'a': [3, 4]})

    # Act & Assert
    helper_test_df_and_result(df, False)


def test_single_column_dataset_single_value():
    # Arrange
    df = pd.DataFrame({'a': ['b', 'b']})

    # Act & Assert
    helper_test_df_and_result(df, True)


def test_multi_column_dataset_single_value():
    # Arrange
    df = pd.DataFrame({'a': ['b', 'b', 'b'], 'b': ['a', 'a', 'a'], 'f': [1, 2, 3]})

    # Act & Assert
    helper_test_df_and_result(df, True)


def test_multi_column_dataset_single_value_with_ignore():
    # Arrange
    df = pd.DataFrame({'a': ['b', 'b', 'b'], 'b': ['a', 'a', 'a'], 'f': [1, 2, 3]})

    # Act & Assert
    helper_test_df_and_result(df, False, ignore_columns=['a', 'b'])


def test_empty_df_single_value():
    #Arrange
    df = pd.DataFrame()

    # Act & Assert
    helper_test_df_and_result(df, False)


def test_single_value_object(iris_dataset):
    # Arrange
    sv = IsSingleValue()

    # Act
    result = sv.run(iris_dataset)

    # Assert
    assert_that(not result.value)


def test_single_value_ignore_columns():
    # Arrange
    sv = IsSingleValue(ignore_columns=['b', 'a'])
    df = pd.DataFrame({'a': ['b', 'b', 'b'], 'b':['a', 'a', 'a'], 'f':[1, 2, 3]})

    # Act
    result = sv.run(df)

    # Assert
    assert_that(not result.value)


def test_single_value_ignore_column():
    # Arrange
    sv = IsSingleValue(ignore_columns='bbb')
    df = pd.DataFrame({'a': ['b', 'b', 'b'], 'bbb':['a', 'a', 'a'], 'f':[1, 2, 3]})

    # Act
    result = sv.run(df)

    # Assert
    assert_that(result.value)


def test_wrong_ignore_columns_single_value():
    # Arrange
    df = pd.DataFrame({'a': ['b', 'b', 'b'], 'bbb':['a', 'a', 'a'], 'f':[1, 2, 3]})

    # Act
    cls = IsSingleValue(ignore_columns=['bbb', 'd'])
    assert_that(calling(cls.run).with_args(df),
                raises(MLChecksValueError, 'Given columns do not exist in dataset: d'))


def test_wrong_input_single_value():
    # Act
    cls = IsSingleValue(ignore_columns=['bbb', 'd'])

    assert_that(calling(cls.run).with_args('some string'),
                raises(MLChecksValueError, 'dataset must be of type DataFrame or Dataset, but got: str'))
