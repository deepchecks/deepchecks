"""Tests for Invalid Chars check"""
import pandas as pd

from hamcrest import assert_that, has_length, calling, raises
from mlchecks.checks.integrity.special_chars import special_characters
from mlchecks.utils import MLChecksValueError


def test_single_column_no_invalid():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = special_characters(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_single_column_invalid():
    # Arrange
    data = {'col1': [1, 'bar!', 'cat', '#@$%']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = special_characters(dataframe)
    # Assert
    assert_that(result.value, has_length(1))


def test_single_column_multi_invalid():
    # Arrange
    data = {'col1': ['1', 'bar!', 'ca\nt', '\n ']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = special_characters(dataframe)
    # Assert
    assert_that(result.value, has_length(1))


def test_double_column_one_invalid():
    # Arrange
    data = {'col1': ['1', 'BAR', '!!!', 'hey-oh'], 'col2': ['', 6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = special_characters(dataframe)
    # Assert
    assert_that(result.value, has_length(1))


def test_double_column_ignored_invalid():
    # Arrange
    data = {'col1': ['1', 'bar!', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = special_characters(dataframe, ignore_columns=['col1'])
    # Assert
    assert_that(result.value, has_length(0))


def test_double_column_specific_invalid():
    # Arrange
    data = {'col1': ['1', 'bar^', '^?!'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = special_characters(dataframe, columns=['col1'])
    # Assert
    assert_that(result.value, has_length(1))


def test_double_column_specific_and_ignored_invalid():
    # Arrange
    data = {'col1': ['1', 'bar()', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act & Assert
    assert_that(calling(special_characters).with_args(dataframe, ignore_columns=['col1'], columns=['col1']),
                raises(MLChecksValueError))


def test_double_column_double_invalid():
    # Arrange
    data = {'col1': ['1_', 'bar', 'cat}', '{}'], 'col2': ['&!', 6, '66&.66.6', 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = special_characters(dataframe)
    # Assert
    assert_that(result.value, has_length(2))
