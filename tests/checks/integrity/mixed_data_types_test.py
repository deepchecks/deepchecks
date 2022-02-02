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
"""Tests for Mixed Types check"""
import numpy as np
import pandas as pd

# Disable wildcard import check for hamcrest
from hamcrest import assert_that, has_length, calling, raises, has_items, has_entry, has_entries, close_to

from deepchecks.core import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks.integrity.mixed_data_types import MixedDataTypes

from tests.checks.utils import equal_condition_result


def test_single_column_no_mix():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedDataTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_single_column_explicit_mix():
    # Arrange
    data = {'col1': [1, 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedDataTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_entries({
        'strings': close_to(0.66, 0.01), 'numbers': close_to(0.33, 0.01)
    })))


def test_single_column_stringed_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedDataTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_entries({
        'strings': close_to(0.66, 0.01), 'numbers': close_to(0.33, 0.01)
    })))


def test_double_column_one_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedDataTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_entries({
        'strings': close_to(0.66, 0.01), 'numbers': close_to(0.33, 0.01)
    })))


def test_double_column_ignored_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedDataTypes(ignore_columns=['col1']).run(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_double_column_specific_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': ['6', 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedDataTypes(columns=['col1']).run(dataframe)
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
    check = MixedDataTypes(ignore_columns=['col1'], columns=['col1'])
    assert_that(calling(check.run).with_args(dataframe),
                raises(DeepchecksValueError))


def test_double_column_double_mix():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, '66.66.6', 666.66]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedDataTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_entries({
        'col1': has_entries({'strings': close_to(0.66, 0.01), 'numbers': close_to(0.33, 0.01)}),
        'col2': has_entries({'strings': close_to(0.33, 0.01), 'numbers': close_to(0.66, 0.01)})
    }))


def test_condition_pass_all_columns():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    check = MixedDataTypes().add_condition_rare_type_ratio_not_in_range()
    # Act
    result = check.conditions_decision(check.run(dataframe))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True, name='Rare data types in column are either more than 10% or less '
                                                  'than 1% of the data')
    ))


def test_condition_pass_fail_single_column():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    check = MixedDataTypes(columns=['col1']).add_condition_rare_type_ratio_not_in_range((0.01, 0.4))
    # Act
    result = check.conditions_decision(check.run(dataframe))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Rare data types in column are either more than 40% or less '
                                    'than 1% of the data',
                               details='Found columns with non-negligible quantities of samples with a different '
                                       'data type from the majority of samples: [\'col1\']',
                               category=ConditionCategory.WARN)
    ))


def test_condition_pass_fail_ignore_column():
    # Arrange
    data = {'col1': ['1', 'bar', 'cat'], 'col2': [6, 66, 666.66]}
    dataframe = pd.DataFrame(data=data)
    check = MixedDataTypes(ignore_columns=['col2']).add_condition_rare_type_ratio_not_in_range((0.01, 0.4))
    # Act
    result = check.conditions_decision(check.run(dataframe))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Rare data types in column are either more than 40% or'
                                    ' less than 1% of the data',
                               details='Found columns with non-negligible quantities of samples with a different '
                                       'data type from the majority of samples: [\'col1\']',
                               category=ConditionCategory.WARN)
    ))


def test_fi_n_top(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    train = Dataset(train.data.copy(), label='target', cat_features=['sex'])
    train.data.loc[train.data.index % 4 == 1, 'age'] = 'a'
    train.data.loc[train.data.index % 4 == 1, 'bmi'] = 'a'
    train.data.loc[train.data.index % 4 == 1, 'bp'] = 'a'
    train.data.loc[train.data.index % 4 == 1, 'sex'] = 'a'
    # Arrange
    check = MixedDataTypes(n_top_columns=3)
    # Act
    result = check.run(train, clf)
    # Assert - Display table is transposed so check columns length
    assert_that(result.display[1].columns, has_length(3))


def test_no_mix_nan():
    # Arrange
    data = {'col1': [np.nan, 'bar', 'cat'], 'col2': ['a', np.nan, np.nan]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedDataTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_length(0))


def test_mix_nan():
    # Arrange
    data = {'col1': [np.nan, '1', 'cat'], 'col2': ['7', np.nan, np.nan]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedDataTypes().run(dataframe)
    # Assert
    assert_that(result.value, has_length(1))
