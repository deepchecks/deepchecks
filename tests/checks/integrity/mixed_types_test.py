"""Tests for Mixed Types check"""
import pandas as pd

# Disable wildcard import check for hamcrest
#pylint: disable=unused-wildcard-import,wildcard-import
from hamcrest import assert_that, has_length, calling, raises, has_items, has_entry, has_entries, close_to

from mlchecks.checks.integrity.mixed_types import MixedTypes
from mlchecks.utils import MLChecksValueError
from tests.checks.utils import equal_condition_result


def test_single_column_no_mix():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_single_column_explicit_mix():
    # Arrange
    data = {'col1': [1, 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_entries({
        'strings': close_to(0.66, 0.01), 'numbers': close_to(0.33, 0.01)
    })))


def test_single_column_stringed_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_entries({
        'strings': close_to(0.66, 0.01), 'numbers': close_to(0.33, 0.01)
    })))


def test_double_column_one_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_entries({
        'strings': close_to(0.66, 0.01), 'numbers': close_to(0.33, 0.01)
    })))


def test_double_column_ignored_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes(ignore_columns=['col1']).run(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_double_column_specific_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': ['6', 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes(columns=['col1']).run(dataframe)
    # Assert
    assert_that(result.value, has_length(1))
    assert_that(result.value, has_entry('col1', has_entries({
        'strings': close_to(0.66, 0.01), 'numbers': close_to(0.33, 0.01)
    })))


def test_double_column_specific_and_ignored_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act & Assert
    check = MixedTypes(ignore_columns=['col1'], columns=['col1'])
    assert_that(calling(check.run).with_args(dataframe),
                raises(MLChecksValueError))


def test_double_column_double_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, '66.66.6', 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_entries({
        'col1': has_entries({'strings': close_to(0.66, 0.01), 'numbers': close_to(0.33, 0.01)}),
        'col2': has_entries({'strings': close_to(0.33, 0.01), 'numbers': close_to(0.66, 0.01)})
    }))


def test_condition_pass_all_columns():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    check = MixedTypes().add_condition_type_ratio_no_less_than(0.1)
    # Act
    result = check.conditions_decision(check.run(dataframe))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True, name='Any type ratio is not lower than 0.1 for all columns')
    ))


def test_condition_pass_fail_single_column():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    check = MixedTypes(columns=['col1']).add_condition_type_ratio_no_less_than(0.4)
    # Act
    result = check.conditions_decision(check.run(dataframe))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Any type ratio is not lower than 0.4 for columns: col1',
                               details='Found columns with low type ratio: col1')
    ))


def test_condition_pass_fail_ignore_column():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    check = MixedTypes(ignore_columns=['col2']).add_condition_type_ratio_no_less_than(0.4)
    # Act
    result = check.conditions_decision(check.run(dataframe))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Any type ratio is not lower than 0.4 for all columns ignoring: col2',
                               details='Found columns with low type ratio: col1')
    ))
