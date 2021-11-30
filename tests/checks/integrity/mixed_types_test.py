"""Tests for Mixed Types check"""
import numpy as np
import pandas as pd

# Disable wildcard import check for hamcrest
from hamcrest import assert_that, has_length, calling, raises, has_items, has_entry, has_entries, close_to
from deepchecks.base import Dataset

from deepchecks.checks.integrity.mixed_types import MixedTypes
from deepchecks.errors import DeepchecksValueError
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
                raises(DeepchecksValueError))


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
    check = MixedTypes().add_condition_rare_type_ratio_not_less_than(0.1)
    # Act
    result = check.conditions_decision(check.run(dataframe))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True, name='Rare type ratio is not less than 10.00% of samples in all columns')
    ))


def test_condition_pass_fail_single_column():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    check = MixedTypes(columns=['col1']).add_condition_rare_type_ratio_not_less_than(0.4)
    # Act
    result = check.conditions_decision(check.run(dataframe))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Rare type ratio is not less than 40.00% of samples in columns: col1',
                               details='Found columns with low type ratio: col1')
    ))


def test_condition_pass_fail_ignore_column():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    check = MixedTypes(ignore_columns=['col2']).add_condition_rare_type_ratio_not_less_than(0.4)
    # Act
    result = check.conditions_decision(check.run(dataframe))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Rare type ratio is not less than 40.00% of samples in all columns ignoring: col2',
                               details='Found columns with low type ratio: col1')
    ))


def test_fi_n_top(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[train.data.index % 4 == 1, 'age'] = 'a'
    train.data.loc[train.data.index % 4 == 1, 'bmi'] = 'a'
    train.data.loc[train.data.index % 4 == 1, 'bp'] = 'a'
    train.data.loc[train.data.index % 4 == 1, 'sex'] = 'a'
    # Arrange
    check = MixedTypes(n_top_columns=3)
    # Act
    result = check.run(train, clf)
    # Assert - Display table is transposed so check columns length
    assert_that(result.display[0].columns, has_length(3))


def test_no_mix_nan():
    # Arrange
    data = {'col1': [np.nan, 'bar', 'cat'], 'col2': ['a', np.nan, np.nan]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_mix_nan():
    # Arrange
    data = {'col1': [np.nan, '1', 'cat'], 'col2': ['7', np.nan, np.nan]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_length(1))
