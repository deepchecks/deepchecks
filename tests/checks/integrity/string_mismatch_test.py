"""Contains unit tests for the string_mismatch check."""
import numpy as np
import pandas as pd

from deepchecks.checks import StringMismatch

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


def test_nan():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'earth', 'foo', 'bar', 'dog'],
            'col2': ['SPACE', 'SPACE$$', 'is', 'fun', None, np.nan]}
    df = pd.DataFrame(data=data)
    # Act
    result = StringMismatch().run(df).value
    # Assert - 4 values are mismatch
    assert_that(result, has_length(4))
