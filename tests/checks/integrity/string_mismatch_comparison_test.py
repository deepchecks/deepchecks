"""Contains unit tests for the string_mismatch check."""
import pandas as pd

from mlchecks.checks import string_mismatch_comparison

from hamcrest import assert_that, has_length


def test_single_col_mismatch():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}

    # Act
    result = string_mismatch_comparison(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert - 1 column, 1 baseform are mismatch
    assert_that(result, has_length(1))


def test_mismatch_multi_column():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?'],
            'col2': ['aaa', 'bbb', 'ddd', '><', '123', '111', '444']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep'],
                     'col2': ['aaa!', 'bbb!', 'ddd', '><', '123???', '123!', '__123__', '111']}

    # Act
    result = string_mismatch_comparison(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert - 2 columns, 4 baseforms are mismatch
    assert_that(result, has_length(4))


def test_no_mismatch():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    compared_data = {'col1': ['foo', 'foo', 'bar', 'bar', 'bar', 'dog?!']}

    # Act
    result = string_mismatch_comparison(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_no_mismatch_on_numeric_column():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat'], 'col2': [10, 2.3, 1]}
    compared_data = {'col1': ['foo', 'foo', 'foo'], 'col2': [1, 2.30, 1.0]}

    # Act
    result = string_mismatch_comparison(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_no_mismatch_on_numeric_string_column():
    # Arrange
    data = {'num_str': ['10', '2.3', '1']}
    compared_data = {'num_str': ['1', '2.30', '1.0']}

    # Act
    result = string_mismatch_comparison(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_no_mismatch_on_one_column_numeric():
    # Arrange
    data = {'num_str': ['10', '2.3', '1']}
    compared_data = {'num_str': [1, 2.30, 1.0]}

    # Act
    result = string_mismatch_comparison(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))
