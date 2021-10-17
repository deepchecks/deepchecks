"""Tests for Invalid Chars check"""
import pandas as pd

# Disable wildcard import check for hamcrest
#pylint: disable=unused-wildcard-import,wildcard-import
from hamcrest import *
from mlchecks.checks.integrity.invalid_chars import invalid_chars
from mlchecks.utils import MLChecksValueError


def test_single_column_no_invalid():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = invalid_chars(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_single_column_invalid():
    # Arrange
    data = {'col1': [1, 'bar!', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = invalid_chars(dataframe)
    # Assert
    assert_that(result.value, has_length(1))


def test_single_column_multi_invalid():
    # Arrange
    data = {'col1': ['1', 'bar!', 'ca\nt']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = invalid_chars(dataframe)
    # Assert
    assert_that(result.value, has_length(1))

def test_double_column_one_invalid():
    # Arrange
    data = {'col1': ['1', 'ba!r', 'cat'], 'col2': [ 6,66,666.66 ]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = invalid_chars(dataframe)
    # Assert
    assert_that(result.value, has_length(1))

def test_double_column_ignored_invalid():
    # Arrange
    data = {'col1': ['1', 'bar!', 'cat'], 'col2': [ 6,66,666.66 ]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = invalid_chars(dataframe,ignore_columns=['col1'])
    # Assert
    assert_that(result.value, has_length(0))

def test_double_column_specific_invalid():
    # Arrange
    data = {'col1': ['1', 'bar^', 'cat'], 'col2': [ 6,66,666.66 ]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = invalid_chars(dataframe,columns=['col1'])
    # Assert
    assert_that(result.value, has_length(1))

def test_double_column_specific_and_ignored_invalid():
    # Arrange
    data = {'col1': ['1', 'bar()', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act & Assert
    assert_that(calling(invalid_chars).with_args(dataframe, ignore_columns=['col1'], columns=['col1']),
                raises(MLChecksValueError))


def test_double_column_double_invalid():
    # Arrange
    data = {'col1': ['1_', 'bar', 'cat}'], 'col2': [6, '66&.66.6', 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = invalid_chars(dataframe)
    # Assert
    assert_that(result.value, has_length(2))
