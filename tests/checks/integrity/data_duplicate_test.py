"""Tests for Mixed Nulls check"""
import pandas as pd

from hamcrest import assert_that, close_to, equal_to, calling, raises

from mlchecks.checks.integrity.data_duplicates import data_duplicates, DataDuplicates
from mlchecks.utils import MLChecksValueError


def test_data_duplicates():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    assert_that(data_duplicates(duplicate_data).value, close_to(0.40, 0.01))
    check_obj = DataDuplicates()
    assert_that(check_obj.run(duplicate_data).value, close_to(0.40, 0.01))


def test_data_duplicates_columns():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    assert_that(data_duplicates(duplicate_data, columns=['col1']).value, close_to(0.80, 0.01))
    check_obj = DataDuplicates(columns=['col1'])
    assert_that(check_obj.run(duplicate_data).value, close_to(0.80, 0.01))


def test_data_duplicates_ignore_columns():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    assert_that(data_duplicates(duplicate_data, ignore_columns=['col3']).value, close_to(0.80, 0.01))
    check_obj = DataDuplicates(columns=['col1'])
    assert_that(check_obj.run(duplicate_data).value, close_to(0.80, 0.01))


def test_data_duplicates_n_to_show():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    assert_that(data_duplicates(duplicate_data, n_to_show=2).value, close_to(0.40, 0.01))
    check_obj = DataDuplicates(n_to_show=2)
    assert_that(check_obj.run(duplicate_data).value, close_to(0.40, 0.01))


def test_data_duplicates_no_duplicate():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]})
    assert_that(data_duplicates(duplicate_data).value, equal_to(0))
    check_obj = DataDuplicates()
    assert_that(check_obj.run(duplicate_data).value, equal_to(0))


def test_data_duplicates_empty():
    no_data = pd.DataFrame({'col1': [],
                            'col2': [],
                            'col3': []})
    assert_that(
        calling(data_duplicates).with_args(no_data),
        raises(MLChecksValueError, 'Dataset does not contain any data'))


def test_data_duplicates_ignore_index_column():
    duplicate_data = pd.DataFrame({'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                   'col3': [2, 3, 4, 4, 4, 3, 4, 5, 6, 4]})
    duplicate_data = duplicate_data.set_index('col3')
    assert_that(data_duplicates(duplicate_data).value, close_to(0.80, 0.01))
    check_obj = DataDuplicates()
    assert_that(check_obj.run(duplicate_data).value, close_to(0.80, 0.01))
