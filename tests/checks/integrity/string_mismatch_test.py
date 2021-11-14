"""Contains unit tests for the string_mismatch check."""
import numpy as np
import pandas as pd

from mlchecks.base import Dataset
from mlchecks.checks import StringMismatch

from hamcrest import assert_that, has_length


def test_double_col_mismatch():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', '$deeP$', 'earth', 'foo', 'bar', 'foo?']}
    df = pd.DataFrame(data=data)
    # Act
    result = StringMismatch().run(df).value
    # Assert - 6 values are mismatch
    assert_that(result, has_length(6))


def test_single_mismatch():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', '$deeP$', 'earth', 'foo', 'bar', 'dog']}
    df = pd.DataFrame(data=data)
    # Act
    result = StringMismatch().run(df).value
    # Assert - 4 values are mismatch
    assert_that(result, has_length(4))


def test_mismatch_multi_column():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'earth', 'foo', 'bar', 'dog'],
            'col2': ['SPACE', 'SPACE$$', 'is', 'fun', 'go', 'moon']}
    df = pd.DataFrame(data=data)
    # Act
    result = StringMismatch().run(df).value
    # Assert - 4 values are mismatch
    assert_that(result, has_length(4))


def test_mismatch_multi_column_ignore():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'earth', 'foo', 'bar', 'dog'],
            'col2': ['SPACE', 'SPACE$$', 'is', 'fun', 'go', 'moon']}
    df = pd.DataFrame(data=data)
    # Act
    result = StringMismatch(ignore_columns=['col2']).run(df).value
    # Assert - 4 values are mismatch
    assert_that(result, has_length(2))


def test_fi_n_top(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[train.data.index % 3 == 2, 'age'] = 'aaa'
    train.data.loc[train.data.index % 3 == 1, 'age'] = 'aaa!!'
    train.data.loc[train.data.index % 3 == 2, 'bmi'] = 'aaa'
    train.data.loc[train.data.index % 3 == 1, 'bmi'] = 'aaa!!'
    train.data.loc[train.data.index % 3 == 2, 'bp'] = 'aaa'
    train.data.loc[train.data.index % 3 == 1, 'bp'] = 'aaa!!'
    train.data.loc[train.data.index % 3 == 2, 'sex'] = 'aaa'
    train.data.loc[train.data.index % 3 == 1, 'sex'] = 'aaa!!'
    
    # Arrange
    check = StringMismatch(n_top_columns=3)
    # Act
    result_ds = check.run(train, clf).value
    # Assert
    assert_that(result_ds, has_length(3))


def test_nan():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'earth', 'foo', 'bar', 'dog'],
            'col2': ['SPACE', 'SPACE$$', 'is', 'fun', None, np.nan]}
    df = pd.DataFrame(data=data)
    # Act
    result = StringMismatch().run(df).value
    # Assert - 4 values are mismatch
    assert_that(result, has_length(4))
