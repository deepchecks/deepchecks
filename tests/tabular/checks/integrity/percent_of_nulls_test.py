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
"""Tests for Percent Of Nulls check."""
import numpy as np
import pandas as pd
from hamcrest import *

from deepchecks.tabular.checks.data_integrity import PercentOfNulls
from tests.base.utils import equal_condition_result


def test_percent_of_nulls():
    # Arrange
    df = pd.DataFrame({'foo': ['a', 'b', np.nan, None], 'bar': [None, 'a', 'b', 'a']})
    # Act
    result = PercentOfNulls(aggregation_method="mean").run(df)
    # Assert
    assert_that(result.display, has_length(greater_than(0)))
    assert_that(result.value, has_length(2))
    assert_that(result.value.iloc[0, 0], equal_to(0.5))
    assert_that(result.value.iloc[1, 0], equal_to(0.25))
    assert_that(result.reduce_output(), np.mean(result.value.iloc[:, 0]))


def test_percent_of_nulls_without_display():
    # Arrange
    df = pd.DataFrame({'foo': ['a', 'b', np.nan, None], 'bar': [None, 'a', 'b', 'a']})
    # Act
    result = PercentOfNulls().run(df, with_display=False)
    # Assert
    assert_that(result.display, has_length(equal_to(0)))
    assert_that(result.value, has_length(2))
    assert_that(result.value.iloc[0, 0], equal_to(0.5))
    assert_that(result.value.iloc[1, 0], equal_to(0.25))


def test_percent_of_nulls_with_columns_of_categorical_dtype():
    # Arrange
    t = pd.CategoricalDtype(categories=['b', 'a'])
    df = pd.DataFrame({'foo': ['a', 'b', np.nan, None], 'bar': [None, 'a', 'b', 'a']}, dtype=t)
    # Act
    result = PercentOfNulls().run(df)
    # Assert
    assert_that(result.value, has_length(2))
    assert_that(result.value.iloc[0, 0], equal_to(0.5))
    assert_that(result.value.iloc[1, 0], equal_to(0.25))
    assert_that(result.reduce_output(), max(result.value.iloc[:, 0]))


def test_reduce_output_method_none():
    # Arrange
    df = pd.DataFrame({'foo': ['a', 'b', np.nan, None], 'bar': [None, 'a', 'b', 'a']})
    # Act
    result = PercentOfNulls(aggregation_method='none').run(df)
    # Assert
    assert_that(result.value, has_length(2))
    assert_that(result.value.iloc[0, 0], equal_to(0.5))
    assert_that(result.value.iloc[1, 0], equal_to(0.25))
    assert_that(result.reduce_output(), has_entries({'foo': 0.5, 'bar': 0.25}))


def test_exclude_parameter():
    # Arrange
    df = pd.DataFrame({'foo': ['a', 'b', np.nan, None], 'bar': [None, 'a', 'b', 'a']})
    # Act
    result = PercentOfNulls(ignore_columns=['foo']).run(df)
    # Assert
    assert_that(result.value, has_length(1))
    assert_that(result.value.iloc[0, 0], equal_to(0.25))


def test_columns_parameter():
    # Arrange
    df = pd.DataFrame({'foo': ['a', 'b', np.nan, None], 'bar': [None, 'a', 'b', 'a']})
    # Act
    result = PercentOfNulls(columns=['foo']).run(df)
    # Assert
    assert_that(result.value, has_length(1))
    assert_that(result.value.iloc[0, 0], equal_to(0.5))


def test_condition():
    # Arrange
    df = pd.DataFrame({'foo': ['a', 'b'], 'bar': ['a', 'a']})
    check = PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0.01)
    # Act
    check_result = check.run(df)
    conditions_results = check.conditions_decision(check_result)

    assert_that(conditions_results, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 2 relevant columns',
                               name='Percent of null values in each column is not greater than 1%')
    ))


def test_not_passing_condition():
    # Arrange
    df = pd.DataFrame({'foo': ['a', 'b', np.nan, None], 'bar': [None, 'a', 'b', 'a']})
    check = PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0.01)
    # Act
    check_result = check.run(df)
    conditions_results = check.conditions_decision(check_result)

    assert_that(conditions_results, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 2 columns with ratio of nulls above threshold: '
                                       '\n{\'foo\': \'50%\', \'bar\': \'25%\'}',
                               name='Percent of null values in each column is not greater than 1%')
    ))
