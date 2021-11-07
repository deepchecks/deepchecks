"""Tests for Invalid Chars check"""
import pandas as pd

from hamcrest import assert_that, has_length, calling, raises, has_items
from mlchecks.checks.integrity.special_chars import SpecialCharacters
from mlchecks.utils import MLChecksValueError


def test_single_column_no_invalid():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_single_column_invalid():
    # Arrange
    data = {'col1': [1, 'bar!', 'cat', '#@$%']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, has_length(1))
    assert_that(result.value.iloc[0]['Most Common Special-Only Samples'], has_items('#@$%'))


def test_single_column_multi_invalid():
    # Arrange
    data = {'col1': ['1', 'bar!', 'ca\nt', '\n ']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, has_length(1))


def test_double_column_one_invalid():
    # Arrange
    data = {'col1': ['^', '?!', '!!!', '?!', '!!!', '?!'], 'col2': ['', 6, 66, 666.66, 7, 5]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, has_length(1))
    assert_that(result.value.iloc[0]['Most Common Special-Only Samples'], has_items('!!!', '?!'))


def test_double_column_ignored_invalid():
    # Arrange
    data = {'col1': ['1', 'bar!', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters(ignore_columns=['col1']).run(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_double_column_specific_invalid():
    # Arrange
    data = {'col1': ['1', 'bar^', '^?!'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters(columns=['col1']).run(dataframe)
    # Assert
    assert_that(result.value, has_length(1))
    assert_that(result.value.iloc[0]['Most Common Special-Only Samples'], has_items('^?!'))


def test_double_column_specific_and_ignored_invalid():
    # Arrange
    data = {'col1': ['1', 'bar()', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act & Assert
    check = SpecialCharacters(ignore_columns=['col1'], columns=['col1'])
    assert_that(calling(check.run).with_args(dataframe),
                raises(MLChecksValueError))


def test_double_column_double_invalid():
    # Arrange
    data = {'col1': ['1_', 'bar', 'cat}', '{}'], 'col2': ['&!', 6, '66&.66.6', 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, has_length(2))
    assert_that(result.value.loc['col1']['Most Common Special-Only Samples'], has_items('{}'))
    assert_that(result.value.loc['col2']['Most Common Special-Only Samples'], has_items('&!'))
