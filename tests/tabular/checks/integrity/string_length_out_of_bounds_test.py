# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Contains unit tests for the string_length_out_of_bounds check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, equal_to, has_entries, has_entry, has_items, has_length

from deepchecks.core import ConditionCategory
from deepchecks.tabular.checks import StringLengthOutOfBounds
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_no_outliers():
    # Arrange
    col_data = ['a', 'b'] * 100
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = StringLengthOutOfBounds().run(df).value
    # Assert
    assert_that(result, equal_to({'col1': {'outliers': []}}))


def test_single_outlier():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcd'*1000)
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = StringLengthOutOfBounds().run(df).value
    # Assert
    assert_that(result, has_entries(col1=has_entry('outliers', has_length(1))))


def test_outlier_skip_categorical_column():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcd'*1000)
    data = {'categorical': ['hi']*201,
            'col2': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = StringLengthOutOfBounds().run(df).value
    # Assert
    assert_that(result, has_entries(col2=has_entry('outliers', has_length(1))))


def test_outlier_multiple_outliers():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcdefgh')
    col_data.append('abcdefgh')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = StringLengthOutOfBounds().run(df).value
    # Assert
    assert_that(result, has_entries(col1=has_entries(outliers=has_length(1))))


def test_outlier_multiple_outlier_ranges():
    # Arrange
    col_data = ['abcdefg', 'efghabc'] * 100
    col_data.append('a')
    col_data.append('abcdbcdbcdbabcd')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = StringLengthOutOfBounds().run(df).value
    # Assert
    assert_that(result, has_entries(col1=has_entries(outliers=equal_to(
        [{'range': {'min': 1, 'max': 1}, 'n_samples': 1}, {'range': {'min': 15, 'max': 15}, 'n_samples': 1}]
    ))))


def test_fi_n_top(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[0, 'age'] = 'aaa' * 1000
    train.data.loc[0, 'bmi'] = 'aaa' * 1000
    train.data.loc[0, 'bp'] = 'aaa' * 1000
    train.data.loc[0, 'sex'] = 'aaa' * 1000
    # Arrange
    check = StringLengthOutOfBounds(n_top_columns=3)
    # Act
    result_ds = check.run(train, clf).display[1]
    # Assert
    assert_that(result_ds, has_length(3))


def test_nan():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcdefg')
    col_data.append('abcdefg')
    col_data.append(np.nan)
    col_data.append(np.nan)
    col_data.append(np.nan)
    col_data.append(np.nan)
    col_data.append(np.nan)
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    result = StringLengthOutOfBounds().run(df).value
    # Assert
    assert_that(result, has_entries(col1=has_entries(outliers=has_length(1), n_samples=202)))


def test_condition_count_fail():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcdefg')
    col_data.append('abcdefg')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    check = StringLengthOutOfBounds().add_condition_number_of_outliers_not_greater_than(1)

    # Act
    result = check.conditions_decision(check.run(df))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 1 out of 1 columns with number of outliers above threshold: '
                                       '{\'col1\': 2}',
                               name='Number of outliers not greater than 1 string length outliers')
    ))


def test_condition_count_pass():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcdefg')
    col_data.append('abcdefg')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    check = StringLengthOutOfBounds().add_condition_number_of_outliers_not_greater_than(10)

    # Act
    result = check.conditions_decision(check.run(df))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 1 columns',
                               name='Number of outliers not greater than 10 string length outliers')
    ))


def test_condition_ratio_fail():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcdefg')
    col_data.append('abcdefg')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    check = StringLengthOutOfBounds().add_condition_ratio_of_outliers_not_greater_than(0.001)

    # Act
    result = check.conditions_decision(check.run(df))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 1 out of 1 relevant columns with outliers ratio above threshold: '
                                       '{\'col1\': \'0.99%\'}',
                               name='Ratio of outliers not greater than 0.1% string length outliers',
                               category=ConditionCategory.WARN)
    ))


def test_condition_ratio_pass():
    # Arrange
    col_data = ['a', 'b'] * 100
    col_data.append('abcdefg')
    col_data.append('abcdefg')
    data = {'col1': col_data}
    df = pd.DataFrame(data=data)
    # Act
    check = StringLengthOutOfBounds().add_condition_ratio_of_outliers_not_greater_than(0.1)

    # Act
    result = check.conditions_decision(check.run(df))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 1 relevant column',
                               name='Ratio of outliers not greater than 10% string length outliers')
    ))


def test_condition_pass_on_no_outliers():
    # Arrange
    col_data = ['a', 'b'] * 100
    df = pd.DataFrame(data={'col1': col_data})
    check = StringLengthOutOfBounds().add_condition_ratio_of_outliers_not_greater_than(0)
    # Act
    result = check.run(df)
    # Assert
    assert_that(result.conditions_results, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 1 relevant column',
                               name='Ratio of outliers not greater than 0% string length outliers')
    ))
