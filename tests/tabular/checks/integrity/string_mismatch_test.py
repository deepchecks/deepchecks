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
"""Contains unit tests for the string_mismatch check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, greater_than, has_entries, has_entry, has_items, has_length

from deepchecks.core import ConditionCategory
from deepchecks.tabular.checks.data_integrity import StringMismatch
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_double_col_mismatch():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', '$deeP$', 'earth', 'foo', 'bar', 'foo?']}
    df = pd.DataFrame(data=data)
    # Act
    result = StringMismatch().run(df)
    # Assert
    assert_that(result.value, has_entry('col1', has_entries({
        'deep': has_length(4), 'foo': has_length(2)
    })))
    assert_that(result.display, has_length(greater_than(0)))


def test_double_col_mismatch_without_display():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', '$deeP$', 'earth', 'foo', 'bar', 'foo?']}
    df = pd.DataFrame(data=data)
    # Act
    result = StringMismatch().run(df, with_display=False)
    # Assert
    assert_that(result.value, has_entry('col1', has_entries({
        'deep': has_length(4), 'foo': has_length(2)
    })))
    assert_that(result.display, has_length(0))


def test_single_mismatch():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', '$deeP$', 'earth', 'foo', 'bar', 'dog']}
    df = pd.DataFrame(data=data)
    # Act
    result = StringMismatch().run(df).value
    # Assert
    assert_that(result, has_entry('col1', has_entry('deep', has_length(4))))


def test_mismatch_multi_column():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'earth', 'foo', 'bar', 'dog'],
            'col2': ['SPACE', 'SPACE$$', 'is', 'fun', 'go', 'moon']}
    df = pd.DataFrame(data=data)
    # Act
    result = StringMismatch().run(df).value
    # Assert
    assert_that(result, has_entries({
        'col1': has_entry('deep', has_length(2)),
        'col2': has_entry('space', has_length(2))
    }))


def test_mismatch_multi_column_ignore():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'earth', 'foo', 'bar', 'dog'],
            'col2': ['SPACE', 'SPACE$$', 'is', 'fun', 'go', 'moon']}
    df = pd.DataFrame(data=data)
    # Act
    result = StringMismatch(ignore_columns=['col2']).run(df).value
    # Assert
    assert_that(result, has_length(1))
    assert_that(result, has_entry('col1', has_entry('deep', has_length(2))))


def test_condition_no_more_than_fail():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', '$deeP$', 'earth', 'foo', 'bar', 'foo?']}
    df = pd.DataFrame(data=data)
    check = StringMismatch().add_condition_number_variants_less_or_equal(2)
    # Act
    result = check.conditions_decision(check.run(df))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(
            is_pass=False,
            name='Number of string variants is less or equal to 2',
            details='Found 1 out of 1 columns with amount of variants above threshold: '
                    '{\'col1\': [\'deep\']}',
            category=ConditionCategory.WARN)
    ))


def test_condition_no_more_than_pass():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', '$deeP$', 'earth', 'foo', 'bar', 'foo?']}
    df = pd.DataFrame(data=data)
    check = StringMismatch().add_condition_number_variants_less_or_equal(4)
    # Act
    result = check.conditions_decision(check.run(df))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 1 relevant column',
                               name='Number of string variants is less or equal to 4')
    ))


def test_condition_no_variants_fail():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', '$deeP$', 'earth', 'foo', 'bar', 'foo?']}
    df = pd.DataFrame(data=data)
    check = StringMismatch().add_condition_no_variants()
    # Act
    result = check.conditions_decision(check.run(df))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(
            is_pass=False,
            name='No string variants',
            details='Found 1 out of 1 columns with amount of variants above threshold: '
                    '{\'col1\': [\'deep\', \'foo\']}',
            category=ConditionCategory.WARN)
    ))


def test_condition_no_variants_pass():
    # Arrange
    data = {'col1': ['Deep', 'high', 'low!!!', '$shallow$', 'mild', 'foo', 'bar']}
    df = pd.DataFrame(data=data)
    check = StringMismatch().add_condition_no_variants()
    # Act
    result = check.conditions_decision(check.run(df))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 1 relevant column',
                               name='No string variants')
    ))


def test_condition_percent_variants_no_more_than_fail():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', '$deeP$', 'earth', 'foo', 'bar', 'foo?']}
    df = pd.DataFrame(data=data)
    check = StringMismatch().add_condition_ratio_variants_less_or_equal(0.1)
    # Act
    result = check.conditions_decision(check.run(df))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Ratio of variants is less or equal to 10%',
                               details='Found 1 out of 1 relevant columns with variants ratio above threshold: '
                                       '{\'col1\': \'75%\'}')
    ))


def test_condition_percent_variants_no_more_than_pass():
    # Arrange
    data = {'col1': ['Deep', 'shallow', 'high!!!', '$deeP$', 'earth', 'foo', 'bar', 'foo?']}
    df = pd.DataFrame(data=data)
    check = StringMismatch().add_condition_ratio_variants_less_or_equal(0.5)
    # Act
    result = check.conditions_decision(check.run(df))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 1 relevant column',
                               name='Ratio of variants is less or equal to 50%')
    ))


def test_fi_n_top(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[train.data.index % 3 == 2, 'age'] = 'aaa'
    train.data.loc[train.data.index % 3 == 1, 'age'] = 'aaa!!'
    train.data.loc[train.data.index % 3 == 2, 'bmi'] = 'aaa'
    train.data.loc[train.data.index % 3 == 1, 'bmi'] = 'aaa!!'
    train.data.loc[train.data.index % 3 == 2, 'bp'] = 'aaa'
    train.data.loc[train.data.index % 3 == 1, 'bp'] = 'aaa!!'
    train.data.loc[train.data.index % 3 == 2, 'sex'] = 'aaa'
    train.data.loc[train.data.index % 3 == 1, 'sex'] = 'aaa!!'

    # Arrange
    check = StringMismatch(n_top_columns=3)
    # Act
    result = check.run(train)
    # Assert
    assert_that(result.display[1], has_length(3))


def test_nan():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'earth', 'foo', 'bar', 'dog'],
            'col2': ['SPACE', 'SPACE$$', 'is', 'fun', None, np.nan]}
    df = pd.DataFrame(data=data)
    # Act
    result = StringMismatch().run(df).value
    # Assert
    assert_that(result, has_entries({
        'col1': has_length(1),
        'col2': has_length(1)
    }))


def test_invalid_column():
    data = {'col1': [pd.Timestamp(1), pd.Timestamp(2000000), 'Deep', 'deep', 'earth', 'foo', 'bar', 'dog']}
    df = pd.DataFrame(data=data)
    # Act
    result = StringMismatch().run(df).value
    # Assert
    assert_that(result, has_length(0))
