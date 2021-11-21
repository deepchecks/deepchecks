"""Tests for Single Value Check"""
import numpy as np
import pandas as pd
from deepchecks.utils import DeepchecksValueError
from deepchecks.checks.integrity.is_single_value import IsSingleValue
from hamcrest import assert_that, calling, raises, equal_to, has_items

from tests.checks.utils import equal_condition_result


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
                raises(DeepchecksValueError, 'Given columns do not exist in dataset: d'))


def test_wrong_input_single_value():
    # Act
    cls = IsSingleValue(ignore_columns=['bbb', 'd'])

    assert_that(calling(cls.run).with_args('some string'),
                raises(DeepchecksValueError, 'dataset must be of type DataFrame or Dataset, but got: str'))


def test_nans(df_with_fully_nan, df_with_single_nan_in_col):
    # Arrange
    sv = IsSingleValue()

    # Act
    full_result = sv.run(df_with_fully_nan)
    single_result = sv.run(df_with_single_nan_in_col)

    # Assert
    assert_that(full_result.value)
    assert_that(not single_result.value)


def test_condition_fail():
    df = pd.DataFrame({'a': ['b', 'b', 'b'], 'bbb': ['a', 'a', 'a'], 'f': [1, 2, 3]})
    check = IsSingleValue().add_condition_not_single_value()
    result = check.conditions_decision(check.run(df))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Columns containing a single value: [\'a\', \'bbb\']',
                               name='columns do not contain only a single value')))


def test_condition_pass():
    df = pd.DataFrame({'a': ['b', 'asadf', 'b'], 'bbb': ['a', 'a', np.nan], 'f': [1, 2, 3]})
    check = IsSingleValue().add_condition_not_single_value()
    result = check.conditions_decision(check.run(df))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='columns do not contain only a single value')))
