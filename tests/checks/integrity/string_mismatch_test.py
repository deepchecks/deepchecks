import pandas as pd

from mlchecks.checks import string_mismatch
# Disable wildcard import check for hamcrest
#pylint: disable=unused-wildcard-import,wildcard-import
from hamcrest import *


def test_double_col_mismatch():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', '$deeP$', 'earth', 'foo', 'bar', 'foo?']}
    df = pd.DataFrame(data=data)
    # Act
    result = string_mismatch(df).value
    # Assert - 6 values are mismatch
    assert_that(result, has_length(6))


def test_single_mismatch():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', '$deeP$', 'earth', 'foo', 'bar', 'dog']}
    df = pd.DataFrame(data=data)
    # Act
    result = string_mismatch(df).value
    # Assert - 4 values are mismatch
    assert_that(result, has_length(4))


def test_mismatch_multi_column():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'earth', 'foo', 'bar', 'dog'],
            'col2': ['SPACE', 'SPACE$$', 'is', 'fun', 'go', 'moon']}
    df = pd.DataFrame(data=data)
    # Act
    result = string_mismatch(df).value
    # Assert - 4 values are mismatch
    assert_that(result, has_length(4))


def test_mismatch_multi_column_ignore():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'earth', 'foo', 'bar', 'dog'],
            'col2': ['SPACE', 'SPACE$$', 'is', 'fun', 'go', 'moon']}
    df = pd.DataFrame(data=data)
    # Act
    result = string_mismatch(df, ignore_columns=['col2']).value
    # Assert - 4 values are mismatch
    assert_that(result, has_length(2))
