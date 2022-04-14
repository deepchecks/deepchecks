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
"""Tests for Mixed Nulls check"""
import pandas as pd
import numpy as np
from hamcrest import assert_that, close_to, equal_to, calling, raises, has_items

from deepchecks.core import ConditionCategory
from deepchecks.core.errors import DatasetValidationError
from deepchecks.tabular.checks.integrity.data_duplicates import DataDuplicates

from tests.checks.utils import equal_condition_result


def test_data_duplicates():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    check_obj = DataDuplicates()
    assert_that(check_obj.run(duplicate_data).value, close_to(0.40, 0.01))


def test_data_duplicates_categorical_dtypes():
    """We used to have a bug when using groupby on category dtypes, this test ensures it doesn't return """
    data = {
        'a': np.random.randint(0, 1000, 300000),
        'b': np.random.randint(0, 1000, 300000),
        'c': np.random.randint(0, 1000, 300000),
        'd': np.random.randint(0, 1000, 300000),
        'e': np.random.randint(0, 1000, 300000)
    }
    duplicate_data = pd.DataFrame(data).astype('category')
    check_obj = DataDuplicates()
    assert_that(check_obj.run(duplicate_data).value, close_to(0.0, 0.001))


def test_data_duplicates_columns():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    check_obj = DataDuplicates(columns=['col1'])
    assert_that(check_obj.run(duplicate_data).value, close_to(0.80, 0.01))


def test_data_duplicates_ignore_columns():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    check_obj = DataDuplicates(columns=['col1'])
    assert_that(check_obj.run(duplicate_data).value, close_to(0.80, 0.01))


def test_data_duplicates_n_to_show():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    check_obj = DataDuplicates(n_to_show=2)
    assert_that(check_obj.run(duplicate_data).value, close_to(0.40, 0.01))


def test_data_duplicates_no_duplicate():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]})
    check_obj = DataDuplicates()
    assert_that(check_obj.run(duplicate_data).value, equal_to(0))


def test_data_duplicates_empty():
    no_data = pd.DataFrame({'col1': [],
                            'col2': [],
                            'col3': []})
    assert_that(
        calling(DataDuplicates().run).with_args(no_data),
        raises(DatasetValidationError, 'dataset cannot be empty'))


def test_data_duplicates_ignore_index_column():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    duplicate_data = duplicate_data.set_index('col3')
    check_obj = DataDuplicates()
    assert_that(check_obj.run(duplicate_data).value, close_to(0.80, 0.01))


def test_anonymous_series():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randint(0,10,(100,3))).reset_index()
    res = DataDuplicates(ignore_columns=['index']).run(df).value
    assert_that(res, close_to(0.05, 0.001))


def test_nan(df_with_nan_row, df_with_single_nan_in_col):
    df = df_with_nan_row.set_index('col2')
    check_obj = DataDuplicates()
    assert_that(check_obj.run(df).value, equal_to(0))

    df = df_with_single_nan_in_col
    check_obj = DataDuplicates()
    assert_that(check_obj.run(df).value, equal_to(0))


def test_condition_fail():
    # Arrange
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    check = DataDuplicates().add_condition_ratio_not_greater_than(0.1)

    # Act
    result = check.conditions_decision(check.run(duplicate_data))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 40% duplicate data',
                               name='Duplicate data ratio is not greater than 10%',
                               category=ConditionCategory.WARN)))


def test_condition():
    # Arrange
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    check = DataDuplicates().add_condition_ratio_not_greater_than()

    # Act
    result = check.conditions_decision(check.run(duplicate_data))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Duplicate data ratio is not greater than 0%')))
