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
"""Contains unit tests for the string_mismatch check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, has_length, has_entry, has_entries, has_items

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import StringMismatchComparison

from tests.checks.utils import equal_condition_result


def test_single_col_mismatch():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert - 1 column, 1 baseform are mismatch
    assert_that(result, has_entry('col1', has_length(1)))


def test_mismatch_multi_column():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?'],
            'col2': ['aaa', 'bbb', 'ddd', '><', '123', '111', '444']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep'],
                     'col2': ['aaa!', 'bbb!', 'ddd', '><', '123???', '123!', '__123__', '111']}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert - 2 columns, 4 baseforms are mismatch
    assert_that(result, has_entries({
        'col1': has_length(1), 'col2': has_length(3)
     }))


def test_no_mismatch():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    compared_data = {'col1': ['foo', 'foo', 'bar', 'bar', 'bar', 'dog?!']}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_no_mismatch_on_numeric_column():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat'], 'col2': [10, 2.3, 1]}
    compared_data = {'col1': ['foo', 'foo', 'foo'], 'col2': [1, 2.30, 1.0]}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_no_mismatch_on_numeric_string_column():
    # Arrange
    data = {'num_str': ['10', '2.3', '1']}
    compared_data = {'num_str': ['1', '2.30', '1.0']}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_no_mismatch_on_one_column_numeric():
    # Arrange
    data = {'num_str': ['10', '2.3', '1']}
    compared_data = {'num_str': [1, 2.30, 1.0]}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_condition_no_new_variants_fail():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}
    check = StringMismatchComparison().add_condition_no_new_variants()
    # Act
    data_df, compared_data_df = pd.DataFrame(data=data), pd.DataFrame(data=compared_data)
    result, *_ = check.conditions_decision(check.run(compared_data_df, data_df))
    # Assert
    assert_that(result, equal_condition_result(
        is_pass=False,
        name='No new variants allowed in test data',
        details='Found columns with ratio of variants above threshold: {\'col1\': \'14.29%\'}'
    ))


def test_condition_no_new_variants_pass():
    # Arrange
    base_data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    tested_data = {'col1': ['Deep', 'deep', 'cat', 'earth', 'foo', 'bar', 'foo?', 'bar']}
    check = StringMismatchComparison().add_condition_no_new_variants()

    # Act
    test_df, base_df = pd.DataFrame(data=tested_data), pd.DataFrame(data=base_data)
    result, *_ = check.conditions_decision(check.run(base_df, test_df))

    # Assert
    assert_that(result, equal_condition_result(
        is_pass=True,
        name='No new variants allowed in test data'
    ))


def test_condition_percent_new_variants_fail():
    # Arrange
    base_data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    tested_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}
    check = StringMismatchComparison().add_condition_ratio_new_variants_not_greater_than(0.1)

    # Act
    test_df, base_df = pd.DataFrame(data=tested_data), pd.DataFrame(data=base_data)
    result, *_ = check.conditions_decision(check.run(base_df, test_df,))

    # Assert
    assert_that(result, equal_condition_result(
        is_pass=False,
        name='Ratio of new variants in test data is not greater than 10%',
        details='Found columns with ratio of variants above threshold: {\'col1\': \'25%\'}'
    ))


def test_condition_percent_new_variants_pass():
    # Arrange
    base_data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    tested_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}
    check = StringMismatchComparison().add_condition_ratio_new_variants_not_greater_than(0.5)
    # Act
    result = check.conditions_decision(check.run(pd.DataFrame(data=tested_data), pd.DataFrame(data=base_data)))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Ratio of new variants in test data is not greater than 50%')
    ))


def test_fi_n_top(diabetes_split_dataset_and_model):
    train, val, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    val = Dataset(val.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[train.data.index % 2 == 0, 'age'] = 'aaa'
    val.data.loc[val.data.index % 2 == 1, 'age'] = 'aaa!!'
    train.data.loc[train.data.index % 2 == 0, 'bmi'] = 'aaa'
    val.data.loc[val.data.index % 2 == 1, 'bmi'] = 'aaa!!'
    train.data.loc[train.data.index % 2 == 0, 'bp'] = 'aaa'
    val.data.loc[val.data.index % 2 == 1, 'bp'] = 'aaa!!'
    train.data.loc[train.data.index % 2 == 0, 'sex'] = 'aaa'
    val.data.loc[val.data.index % 2 == 1, 'sex'] = 'aaa!!'
    # Arrange
    check = StringMismatchComparison(n_top_columns=3)
    # Act
    result = check.run(test_dataset=train, train_dataset=val, model=clf)
    # Assert - The display table is transposed so check length of columns
    assert_that(result.display[1].columns, has_length(3))


def test_nan():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?'],
            'col2': ['aaa', 'bbb', 'ddd', '><', '123', '111', '444']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep'],
                     'col2': ['aaa!', 'bbb!', 'ddd', '><', '123???', '123!', '__123__', np.nan]}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert - 2 columns, 4 baseforms are mismatch
    assert_that(result, has_entries({
        'col1': has_length(1), 'col2': has_length(3)
     }))
