# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Tests for Invalid Chars check"""
import numpy as np
import pandas as pd
from hamcrest import assert_that, calling, equal_to, has_items, has_length, raises

from deepchecks.core import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.data_integrity.special_chars import SpecialCharacters
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_single_column_no_invalid():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, equal_to({'col1': 0}))


def test_single_column_invalid():
    # Arrange
    data = {'col1': [1, 'bar!', 'cat', '#@$%']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, equal_to({'col1': 0.25}))
    assert_that(result.display[1].iloc[0]['Most Common Special-Only Samples'], has_items('#@$%'))


def test_single_column_invalid_without_display():
    # Arrange
    data = {'col1': [1, 'bar!', 'cat', '#@$%']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe, with_display=False)
    # Assert
    assert_that(result.value, equal_to({'col1': 0.25}))
    assert_that(result.display, has_length(0))


def test_single_column_multi_invalid():
    # Arrange
    data = {'col1': ['1', 'bar!', 'ca\nt', '\n ', '!']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, equal_to({'col1': 0.4}))


def test_double_column_one_invalid():
    # Arrange
    data = {'col1': ['^', '?!', '!!!', '?!', '!!!', '?!'], 'col2': ['', 6, 66, 666.66, 7, 5]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, equal_to({'col1': 1, 'col2': 0}))
    assert_that(result.display[1].iloc[0]['Most Common Special-Only Samples'], has_items('!!!', '?!'))


def test_double_column_ignored_invalid():
    # Arrange
    data = {'col1': ['1', 'bar!', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters(ignore_columns=['col1']).run(dataframe)
    # Assert
    assert_that(result.value, equal_to({'col2': 0}))


def test_double_column_specific_invalid():
    # Arrange
    data = {'col1': ['1', 'bar^', '^?!', 'cat'], 'col2': [6, 66, 666.66, 3]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters(columns=['col1']).run(dataframe)
    # Assert
    assert_that(result.value, equal_to({'col1': 0.25}))
    assert_that(result.display[1].iloc[0]['Most Common Special-Only Samples'], has_items('^?!'))


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
    assert_that(result.value, equal_to({'col1': 0.25, 'col2': 0.25}))
    assert_that(result.display[1].loc['col1']['Most Common Special-Only Samples'], has_items('{}'))
    assert_that(result.display[1].loc['col2']['Most Common Special-Only Samples'], has_items('&!'))


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
    result_ds = check.run(train).display[1]
    # Assert
    assert_that(result_ds, has_length(3))


def test_nan():
    # Arrange
    data = {'col1': ['1_', 'bar', 'cat}', '{}', np.nan], 'col2': ['&!', 6, '66&.66.6', 666.66, np.nan]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = SpecialCharacters().run(dataframe)
    # Assert
    assert_that(result.value, equal_to({'col1': 0.2, 'col2': 0.2}))
    assert_that(result.display[1].loc['col1']['Most Common Special-Only Samples'], has_items('{}'))
    assert_that(result.display[1].loc['col2']['Most Common Special-Only Samples'], has_items('&!'))


def test_condition_fail_all(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[train.data.index % 3 == 2, 'age'] = '&!'
    train.data.loc[train.data.index % 3 == 2, 'bmi'] = '&!'
    train.data.loc[train.data.index % 3 == 2, 'bp'] = '&!'
    train.data.loc[train.data.index % 3 == 2, 'sex'] = '&!'
    # Arrange
    check = SpecialCharacters(n_top_columns=3).add_condition_ratio_of_special_characters_less_or_equal()
    # Act
    results = check.conditions_decision(check.run(train))
    # Assert
    assert_that(results, has_items(equal_condition_result(
        is_pass=False,
        name='Ratio of samples containing solely special character is less or equal to 0.1%',
        details='Found 4 out of 11 relevant columns with ratio above threshold: {\'age\': \'34.12%\', \'sex\': '
                '\'34.12%\', \'bmi\': \'34.12%\', \'bp\': \'34.12%\'}',
        category=ConditionCategory.WARN
    )))


def test_condition_fail_some(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[train.data.index % 7 == 2, 'age'] = '&!'
    train.data.loc[train.data.index % 3 == 2, 'bmi'] = '&!'
    train.data.loc[train.data.index % 7 == 2, 'bp'] = '&!'
    train.data.loc[train.data.index % 3 == 2, 'sex'] = '&!'
    # Arrange
    check = SpecialCharacters(n_top_columns=3).add_condition_ratio_of_special_characters_less_or_equal(0.3)
    # Act
    results = check.conditions_decision(check.run(train))
    # Assert
    assert_that(results, has_items(equal_condition_result(
        is_pass=False,
        name='Ratio of samples containing solely special character is less or equal to 30%',
        details='Found 2 out of 11 relevant columns with ratio above threshold: '
                '{\'sex\': \'34.12%\', \'bmi\': \'34.12%\'}',
        category=ConditionCategory.WARN
    )))


def test_condition_pass(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])

    # Arrange
    check = SpecialCharacters(n_top_columns=3).add_condition_ratio_of_special_characters_less_or_equal()
    # Act
    results = check.conditions_decision(check.run(train))
    # Assert
    assert_that(results, has_items(equal_condition_result(
        is_pass=True,
        details='Passed for 11 relevant columns',
        name='Ratio of samples containing solely special character is less or equal to 0.1%',
    )))
