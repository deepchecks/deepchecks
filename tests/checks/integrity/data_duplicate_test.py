"""Tests for Mixed Nulls check"""
import pandas as pd

from hamcrest import assert_that, close_to, equal_to, calling, raises, has_items

from deepchecks.checks.integrity.data_duplicates import DataDuplicates
from deepchecks.errors import DeepchecksValueError
from tests.checks.utils import equal_condition_result


def test_data_duplicates():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    check_obj = DataDuplicates()
    assert_that(check_obj.run(duplicate_data).value, close_to(0.40, 0.01))


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
        raises(DeepchecksValueError, 'Dataset does not contain any data'))


def test_data_duplicates_ignore_index_column():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    duplicate_data = duplicate_data.set_index('col3')
    check_obj = DataDuplicates()
    assert_that(check_obj.run(duplicate_data).value, close_to(0.80, 0.01))


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
                               details='Found 40.00% duplicate data',
                               name='Duplicate data is not greater than 10.00%')))


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
                               name='Duplicate data is not greater than 0%')))
