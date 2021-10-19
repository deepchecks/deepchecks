"""Tests for Mixed Types check"""
import pandas as pd

# Disable wildcard import check for hamcrest
#pylint: disable=unused-wildcard-import,wildcard-import
from hamcrest import assert_that, has_length, calling, raises

from mlchecks.checks.integrity.mixed_types import mixed_types
from mlchecks.utils import MLChecksValueError


def test_single_column_no_mix():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_types(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(0))


def test_single_column_explicit_mix():
    # Arrange
    data = {'col1': [1, 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_types(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(1)) #2 types


def test_single_column_stringed_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_types(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(1))

def test_double_column_one_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_types(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(1))

def test_double_column_ignored_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_types(dataframe,ignore_columns=['col1'])
    # Assert
    assert_that(result.value.columns, has_length(0))

def test_double_column_specific_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_types(dataframe,columns=['col1'])
    # Assert
    assert_that(result.value.columns, has_length(1))

def test_double_column_specific_and_ignored_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act & Assert
    assert_that(calling(mixed_types).with_args(dataframe, ignore_columns=['col1'], columns=['col1']),
                raises(MLChecksValueError))


def test_double_column_double_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, '66.66.6', 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_types(dataframe)
    # Assert
    assert_that(result.value.columns, has_length(2))
