"""Tests for Mixed Nulls check"""
import hamcrest
import numpy as np
import pandas as pd

from hamcrest import assert_that, has_length, has_entry, has_property, has_entries, equal_to, calling, raises

from mlchecks import Dataset, ConditionCategory
from mlchecks.checks.integrity.mixed_nulls import mixed_nulls, MixedNulls
from mlchecks.utils import MLChecksValueError


def test_single_column_no_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_single_column_one_null_type():
    # Arrange
    data = {'col1': ['foo', 'bar', 'null', 'null']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert - Single null type is allowed so return is 0
    assert_that(result.value, has_length(0))


def test_empty_dataframe():
    # Arrange
    data = {'col1': []}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_different_null_types():
    # Arrange
    data = {'col1': [np.NAN, np.NaN, pd.NA, '$$$$$$$$', 'NULL']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(3)))


def test_null_list_param():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat', 'earth', 'earth?', '!E!A!R!T!H', np.NaN, 'null']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe, null_string_list=['earth', 'cat'])
    # Assert
    assert_that(result.value, has_entry('col1', has_length(5)))


def test_check_nan_false_param():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat', 'earth', 'earth?', '!E!A!R!T!H', np.NaN, 'null']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe, null_string_list=['earth'], check_nan=False)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(3)))


def test_single_column_two_null_types():
    # Arrange
    data = {'col1': ['foo', 'bar', 'null', 'nan', 'nan']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(2)))


def test_single_column_different_case_is_count_separately():
    # Arrange
    data = {'col1': ['foo', 'bar', 'Nan', 'nan', 'NaN']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(3)))


def test_single_column_nulls_with_special_characters():
    # Arrange
    data = {'col1': ['', '#@$', 'Nan!', '#nan', '<NaN>']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(5)))


def test_ignore_columns_single():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat'], 'col2': ['nan', 'null', ''], 'col3': [np.NAN, 'none', '3']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe, ignore_columns='col3')
    # Assert - Only col 2 should have results
    assert_that(result.value, has_entry('col2', has_length(3)))


def test_ignore_columns_multi():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat'], 'col2': ['nan', 'null', ''], 'col3': [np.NAN, 'none', '3']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe, ignore_columns=['col3', 'col2'])
    # Assert
    assert_that(result.value, has_length(0))


def test_dataset_no_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat'], 'col2': ['foo', 'bar', 1], 'col3': [1, 2, 3]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_dataset_1_column_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'null'], 'col2': ['foo', 'bar', 1], 'col3': [1, 2, 3]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert - Single null is allowed so still empty return
    assert_that(result.value, has_length(0))


def test_dataset_2_columns_single_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'null'], 'col2': ['Nan', 'bar', 1], 'col3': [1, 2, 3]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = mixed_nulls(dataframe)
    # Assert - Single null is allowed so still empty return
    assert_that(result.value, has_length(0))


def test_condition_max_nulls_not_passed():
    # Arrange
    data = {'col1': ['', '#@$', 'Nan!', '#nan', '<NaN>']}
    dataset = Dataset(pd.DataFrame(data=data))
    check = MixedNulls().add_condition_max_different_nulls(3)

    # Act
    result = check.conditions_decision(check.run(dataset))

    assert_that(result, has_entries({
        'No more than 3 null types for all columns': hamcrest.all_of(
            has_property('is_pass', equal_to(False)),
            has_property('category', ConditionCategory.FAILURE),
            has_property('actual', 'Found columns col1 with more than 3 null types')
        )
    }))


def test_condition_max_nulls_passed():
    # Arrange
    data = {'col1': ['', '#@$', 'Nan!', '#nan', '<NaN>']}
    dataset = Dataset(pd.DataFrame(data=data))
    check = MixedNulls().add_condition_max_different_nulls(10)

    # Act
    result = check.conditions_decision(check.run(dataset))

    assert_that(result, has_entries({
        'No more than 10 null types for all columns': has_property('is_pass', equal_to(True))
    }))


def test_condition_max_nulls_columns_filter():
    # Arrange
    data = {'col1': ['', '#@$', 'Nan!', '#nan', '<NaN>'], 'col2': ['a', 'b', 'c', 'd', 'e']}
    dataset = Dataset(pd.DataFrame(data=data))
    check = MixedNulls().add_condition_max_different_nulls(1, columns=['col2'])

    # Act
    result = check.conditions_decision(check.run(dataset))

    assert_that(result, has_entries({
        'No more than 1 null types for columns: col2': has_property('is_pass', equal_to(True))
    }))


def test_condition_max_nulls_ignore_columns_filter():
    # Arrange
    data = {'col1': ['', '#@$', 'Nan!', '#nan', '<NaN>'], 'col2': ['a', 'b', 'c', 'd', 'e']}
    dataset = Dataset(pd.DataFrame(data=data))
    check = MixedNulls().add_condition_max_different_nulls(1, ignore_columns=['col1'])

    # Act
    result = check.conditions_decision(check.run(dataset))

    assert_that(result, has_entries({
        'No more than 1 null types for all columns ignoring: col1': has_property('is_pass', equal_to(True))
    }))


def test_condition_max_nulls_both_filters_raise_error():
    # Arrange
    check = MixedNulls()

    # Act & Assert
    assert_that(calling(check.add_condition_max_different_nulls).with_args(1, columns=['col1'],
                                                                           ignore_columns=['col2']),
                raises(MLChecksValueError, 'Can not define columns and ignore_columns together'))
