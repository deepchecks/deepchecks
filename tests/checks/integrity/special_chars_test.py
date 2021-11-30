"""Tests for Invalid Chars check"""
import numpy as np
import pandas as pd
from hamcrest import assert_that, has_length, calling, raises, has_items, close_to

from deepchecks.base import Dataset
from deepchecks.checks.integrity.special_chars import SpecialCharacters
from deepchecks.errors import DeepchecksValueError

from tests.checks.utils import equal_condition_result


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
    assert_that(result.value['col1'], close_to(0.25, 0.001))
    assert_that(result.display[0].iloc[0]['Most Common Special-Only Samples'], has_items('#@$%'))


def test_single_column_multi_invalid():
    # Arrange
    data = {'col1': ['1', 'bar!', 'ca\nt', '\n ']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, has_length(1))
    assert_that(result.value['col1'], close_to(0.25, 0.001))


def test_double_column_one_invalid():
    # Arrange
    data = {'col1': ['^', '?!', '!!!', '?!', '!!!', '?!'], 'col2': ['', 6, 66, 666.66, 7, 5]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, has_length(1))
    assert_that(result.value['col1'], close_to(1, 0.001))
    assert_that(result.display[0].iloc[0]['Most Common Special-Only Samples'], has_items('!!!', '?!'))


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
    assert_that(result.value['col1'], close_to(0.333, 0.001))
    assert_that(result.display[0].iloc[0]['Most Common Special-Only Samples'], has_items('^?!'))


def test_double_column_specific_and_ignored_invalid():
    # Arrange
    data = {'col1': ['1', 'bar()', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act & Assert
    check = SpecialCharacters(ignore_columns=['col1'], columns=['col1'])
    assert_that(calling(check.run).with_args(dataframe),
                raises(DeepchecksValueError))


def test_double_column_double_invalid():
    # Arrange
    data = {'col1': ['1_', 'bar', 'cat}', '{}'], 'col2': ['&!', 6, '66&.66.6', 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, has_length(2))
    assert_that(result.value['col1'], close_to(0.25, 0.001))
    assert_that(result.value['col2'], close_to(0.25, 0.001))
    assert_that(result.display[0].loc['col1']['Most Common Special-Only Samples'], has_items('{}'))
    assert_that(result.display[0].loc['col2']['Most Common Special-Only Samples'], has_items('&!'))


def test_fi_n_top(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[train.data.index % 3 == 2, 'age'] = '&!'
    train.data.loc[train.data.index % 3 == 2, 'bmi'] = '&!'
    train.data.loc[train.data.index % 3 == 2, 'bp'] = '&!'
    train.data.loc[train.data.index % 3 == 2, 'sex'] = '&!'
    # Arrange
    check = SpecialCharacters(n_top_columns=3)
    # Act
    result_ds = check.run(train, clf).display[0]
    # Assert
    assert_that(result_ds, has_length(3))


def test_nan():
    # Arrange
    data = {'col1': ['1_', 'bar', 'cat}', '{}', np.nan], 'col2': ['&!', 6, '66&.66.6', 666.66, np.nan]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, has_length(2))
    assert_that(result.display[0].loc['col1']['Most Common Special-Only Samples'], has_items('{}'))
    assert_that(result.display[0].loc['col2']['Most Common Special-Only Samples'], has_items('&!'))


def test_condition_fail_all(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[train.data.index % 3 == 2, 'age'] = '&!'
    train.data.loc[train.data.index % 3 == 2, 'bmi'] = '&!'
    train.data.loc[train.data.index % 3 == 2, 'bp'] = '&!'
    train.data.loc[train.data.index % 3 == 2, 'sex'] = '&!'
    # Arrange
    check = SpecialCharacters(n_top_columns=3).add_condition_ratio_of_special_characters_not_grater_than()
    # Act
    results = check.conditions_decision(check.run(train, clf))
    # Assert
    assert_that(results, has_items(equal_condition_result(
        is_pass=False,
        name='Ratio of entirely special character samples not greater than 0.10% for all columns',
        details='Found columns over threshold ratio: [\'age\', \'sex\', \'bmi\', \'bp\']'
    )))


def test_condition_fail_some(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[train.data.index % 7 == 2, 'age'] = '&!'
    train.data.loc[train.data.index % 3 == 2, 'bmi'] = '&!'
    train.data.loc[train.data.index % 7 == 2, 'bp'] = '&!'
    train.data.loc[train.data.index % 3 == 2, 'sex'] = '&!'
    # Arrange
    check = SpecialCharacters(n_top_columns=3).add_condition_ratio_of_special_characters_not_grater_than(0.3)
    # Act
    results = check.conditions_decision(check.run(train, clf))
    # Assert
    assert_that(results, has_items(equal_condition_result(
        is_pass=False,
        name='Ratio of entirely special character samples not greater than 30.00% for all columns',
        details='Found columns over threshold ratio: [\'sex\', \'bmi\']'
    )))


def test_condition_pass(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])

    # Arrange
    check = SpecialCharacters(n_top_columns=3).add_condition_ratio_of_special_characters_not_grater_than()
    # Act
    results = check.conditions_decision(check.run(train, clf))
    # Assert
    assert_that(results, has_items(equal_condition_result(
        is_pass=True,
        name='Ratio of entirely special character samples not greater than 0.10% for all columns',
    )))