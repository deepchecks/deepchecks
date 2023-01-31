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
"""Tests for Mixed Nulls check"""
import numpy as np
import pandas as pd
from hamcrest import (any_of, assert_that, calling, close_to, equal_to, greater_than, has_entries, has_entry, has_items,
                      has_length, is_, raises)

from deepchecks.core.errors import DatasetValidationError, DeepchecksValueError
from deepchecks.tabular.checks.data_integrity.mixed_nulls import MixedNulls
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_single_column_no_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe)
    # Assert
    assert_that(result.value, equal_to({'col1': {}}))


def test_single_column_one_null_type():
    # Arrange
    data = {'col1': ['foo', 'bar', 'null', 'null']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe)
    assert_that(result.value, equal_to({'col1': {'"null"': {'count': 2, 'percent': 0.5}}}))
    assert_that(result.display, has_length(greater_than(0)))


def test_single_column_one_null_type_without_display():
    # Arrange
    data = {'col1': ['foo', 'bar', 'null', 'null']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe, with_display=False)
    assert_that(result.value, equal_to({'col1': {'"null"': {'count': 2, 'percent': 0.5}}}))
    assert_that(result.display, has_length(0))


def test_empty_dataframe():
    # Arrange
    data = {'col1': []}
    dataframe = pd.DataFrame(data=data)
    # Act
    assert_that(calling(MixedNulls().run).with_args(dataframe),
                raises(DeepchecksValueError, r'Can\'t create a Dataset object with an empty dataframe'))


def test_different_null_types():
    # Arrange
    data = {'col1': [np.NAN, np.NaN, pd.NA, 'value', 'NULL']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(3)))


def test_null_list_param():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat', 'earth', 'earth?', '!E!A!R!T!H', np.nan, 'null']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls(null_string_list=['earth', 'cat']).run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(5)))


def test_check_nan_false_param():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat', 'earth', 'earth?', '!E!A!R!T!H', np.nan, 'null']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls(null_string_list=['earth'], check_nan=False).run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(4)))


def test_single_column_two_null_types():
    # Arrange
    data = {'col1': ['foo', 'bar', 'null', 'nan', 'nan']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(2)))


def test_single_column_different_case_is_count_separately():
    # Arrange
    data = {'col1': ['foo', 'bar', 'Nan', 'nan', 'NaN']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(3)))


def test_numeric_column_nulls():
    # Arrange
    data = {'col1': [1, 2, np.NaN, pd.NA, pd.NaT]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(3)))


def test_numeric_column_nulls_with_none():
    # Arrange
    data = {'col1': [1, 2, np.NaN, pd.NA, pd.NaT, None]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(4)))


def test_mix_value_columns():
    # Arrange
    data = {'col1': [1, 2, np.NaN, pd.NA, pd.NaT, 3], 'col2': ['foo', 'bar', 'Nan', 'nan', 'NaN', None]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(3)))
    assert_that(result.value, has_entry('col2', has_length(4)))


def test_single_column_nulls_with_special_characters():
    # Arrange
    data = {'col1': ['', 'value', 'Nan!', '#nan', '<NaN>']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(4)))


def test_single_column_nulls_only_special_characters():
    # Arrange
    data = {'col1': ['', '!@#$', 'Nan!', '#nan', '<NaN>']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe)
    # Assert
    assert_that(result.value, has_entry('col1', has_length(4)))


def test_ignore_columns_single():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat'], 'col2': ['nan', 'null', ''], 'col3': [np.nan, 'none', '3']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls(ignore_columns='col3').run(dataframe)
    # Assert - Only col 2 should have results
    assert_that(result.value, has_entries(col1=has_length(0), col2=has_length(3)))


def test_ignore_columns_multi():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat'], 'col2': ['nan', 'null', ''], 'col3': [np.nan, 'none', '3']}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls(ignore_columns=['col3', 'col2']).run(dataframe)
    # Assert
    assert_that(result.value, equal_to({'col1': {}}))


def test_dataset_no_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat'], 'col2': ['foo', 'bar', 1], 'col3': [1, 2, 3]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe)
    # Assert
    assert_that(result.value, has_entries(col1=equal_to({}), col2=equal_to({}), col3=equal_to({})))


def test_dataset_1_column_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'null'], 'col2': ['foo', 'bar', 1], 'col3': [1, 2, 3]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe)
    # Assert
    assert_that(result.value, has_entries(col1=has_length(1),
                                          col2=equal_to({}), col3=equal_to({})))


def test_dataset_2_columns_single_nulls():
    # Arrange
    data = {'col1': ['foo', 'bar', 'null'], 'col2': ['Nan', 'bar', 1], 'col3': [1, 2, 3]}
    dataframe = pd.DataFrame(data=data)
    # Act
    result = MixedNulls().run(dataframe)
    # Assert
    assert_that(result.value, has_entries(col1=has_length(1), col2=has_length(1), col3=equal_to({})))


def test_condition_max_nulls_not_passed():
    # Arrange
    data = {'col1': ['', '#@$', 'Nan!', '#nan', '<NaN>']}
    dataset = Dataset(pd.DataFrame(data=data))
    check = MixedNulls().add_condition_different_nulls_less_equal_to(3)

    # Act
    result = check.conditions_decision(check.run(dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Number of different null types is less or equal to 3',
                               details='Found 1 out of 1 columns with amount of null types above threshold: [\'col1\']')
    ))


def test_condition_max_nulls_passed():
    # Arrange
    data = {'col1': ['', '#@$', 'Nan!', '#nan', '<NaN>']}
    dataset = Dataset(pd.DataFrame(data=data))
    check = MixedNulls().add_condition_different_nulls_less_equal_to(10)

    # Act
    result = check.conditions_decision(check.run(dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 1 relevant column',
                               name='Number of different null types is less or equal to 10')
    ))


def test_mixed_nulls_with_categorical_dtype():
    ds = Dataset(pd.DataFrame({
        'foo': pd.Series(['a', 'b', None, None], dtype='category'),
        'bar': [1,2,3,4]
    }))
    assert_that(
        MixedNulls().run(ds).value,
        has_entries({
            'bar': has_length(equal_to(0)),
            'foo': has_entries({
                'numpy.nan': has_entries({
                    'count': equal_to(2),
                    'percent': equal_to(0.5)
                }),
            }),
            # NOTE:
            # * why math.nan, if we see None in foo? *
            # in short, because of pandas null conversion mechanism
            #
            # example:
            # >>> pd.__version__  # 1.3.5
            # >>> s = pd.Series(['a', pd.NA, pd.NaT, np.nan, None],  dtype='category')
            # >>> s.at[1] is np.nan  # True
            # >>> s.at[2] is np.nan  # True
            # >>> s.at[3] is np.nan  # True
        })
    )
