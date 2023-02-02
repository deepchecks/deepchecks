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
"""Tests for Single Value Check"""
import numpy as np
import pandas as pd
from hamcrest import assert_that, calling, equal_to, greater_than, has_items, has_length, raises

from deepchecks.core.errors import DatasetValidationError, DeepchecksValueError
from deepchecks.tabular.checks.data_integrity.is_single_value import IsSingleValue
from tests.base.utils import equal_condition_result


def helper_test_df_and_result(df, expected_result_value,
                              ignore_columns=None, ignore_nan=True,
                              with_display=True):
    # Act
    result = IsSingleValue(ignore_columns=ignore_columns,
                           ignore_nan=ignore_nan).run(df, with_display=with_display)

    # Assert
    assert_that(result.value, equal_to(expected_result_value))
    return result


def test_single_column_dataset_more_than_single_value():
    # Arrange
    df = pd.DataFrame({'a': [3, 4]})

    # Act & Assert
    helper_test_df_and_result(df, {'a': 2})


def test_single_column_dataset_single_value():
    # Arrange
    df = pd.DataFrame({'a': ['b', 'b']})

    # Act & Assert
    res = helper_test_df_and_result(df, {'a': 1})
    assert_that(res.display, has_length(greater_than(0)))


def test_single_column_dataset_single_value_without_display():
    # Arrange
    df = pd.DataFrame({'a': ['b', 'b']})

    # Act & Assert
    res = helper_test_df_and_result(df, {'a': 1}, with_display=False)
    assert_that(res.display, has_length(0))


def test_multi_column_dataset_single_value():
    # Arrange
    df = pd.DataFrame({'a': ['b', 'b', 'b'], 'b': ['a', 'a', 'a'], 'f': [1, 2, 3]})

    # Act & Assert
    helper_test_df_and_result(df, {'a': 1, 'b': 1, 'f': 3})


def test_multi_column_dataset_single_value_with_ignore():
    # Arrange
    df = pd.DataFrame({'a': ['b', 'b', 'b'], 'b': ['a', 'a', 'a'], 'f': [1, 2, 3]})

    # Act & Assert
    helper_test_df_and_result(df, {'f': 3}, ignore_columns=['a', 'b'])


def test_empty_df_single_value():
    # Act & Assert
    assert_that(calling(IsSingleValue().run).with_args(pd.DataFrame()),
                raises(DeepchecksValueError, r'Can\'t create a Dataset object with an empty dataframe'))


def test_single_value_object(iris_dataset):
    # Act & Assert
    helper_test_df_and_result(iris_dataset, {'sepal length (cm)': 35, 'sepal width (cm)': 23, 'petal length (cm)': 43,
                                             'petal width (cm)': 22, 'target': 3})


def test_single_value_ignore_column():
    # Arrange
    df = pd.DataFrame({'a': ['b', 'b', 'b'], 'bbb': ['a', 'a', 'a'], 'f': [1, 2, 3]})

    # Act & Assert
    helper_test_df_and_result(df, {'a': 1, 'f': 3}, ignore_columns='bbb')


def test_wrong_ignore_columns_single_value():
    # Arrange
    df = pd.DataFrame({'a': ['b', 'b', 'b'], 'bbb': ['a', 'a', 'a'], 'f': [1, 2, 3]})

    # Act
    cls = IsSingleValue(ignore_columns=['bbb', 'd'])
    assert_that(calling(cls.run).with_args(df),
                raises(DeepchecksValueError, 'Given columns do not exist in dataset: d'))


def test_wrong_input_single_value():
    # Act
    cls = IsSingleValue(ignore_columns=['bbb', 'd'])

    assert_that(calling(cls.run).with_args('some string'),
            raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str'))


def test_nans(df_with_fully_nan, df_with_single_nan_in_col):
    # Act & Assert
    helper_test_df_and_result(df_with_fully_nan, {'col1': 1, 'col2': 1}, ignore_nan=False)
    helper_test_df_and_result(df_with_single_nan_in_col, {'col1': 11, 'col2': 11}, ignore_nan=False)


def test_ignore_nans(df_with_fully_nan, df_with_single_nan_in_col):
    # Act & Assert
    helper_test_df_and_result(df_with_fully_nan, {'col1': 0, 'col2': 0}, ignore_nan=True)
    helper_test_df_and_result(df_with_single_nan_in_col, {'col1': 10, 'col2': 11}, ignore_nan=True)


def test_condition_fail():
    # Arrange
    df = pd.DataFrame({'a': ['b', 'b', 'b'], 'bbb': ['a', 'a', 'a'], 'f': [1, 2, 3]})
    check = IsSingleValue().add_condition_not_single_value()

    # Act
    result = check.conditions_decision(check.run(df))

    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 2 out of 3 columns with a single value: [\'a\', \'bbb\']',
                               name='Does not contain only a single value')))


def test_condition_pass():
    # Arrange
    df = pd.DataFrame({'a': ['b', 'asadf', 'b'], 'bbb': ['a', 'a', np.nan], 'f': [1, 2, 3]})
    check = IsSingleValue(ignore_nan=False).add_condition_not_single_value()

    # Act
    result = check.conditions_decision(check.run(df))

    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 3 relevant columns',
                               name='Does not contain only a single value')))
