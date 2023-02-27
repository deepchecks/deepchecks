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
"""
Contains unit tests for the data leakage checks
"""
from datetime import datetime

import numpy as np
import pandas as pd
from hamcrest import assert_that, calling, close_to, equal_to, greater_than, has_items, has_length, raises

from deepchecks.core.errors import DatasetValidationError, DeepchecksValueError
from deepchecks.tabular.checks.train_test_validation import DateTrainTestLeakageDuplicates, DateTrainTestLeakageOverlap
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def dataset_from_dict(d: dict, date_name: str = None) -> Dataset:
    dataframe = pd.DataFrame(data=d)
    return Dataset(dataframe, datetime_name=date_name)


def test_dates_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 5, 0, 0)
    ]}, 'col1')
    val_ds = dataset_from_dict({'col1': [
        datetime(2021, 9, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 9, 0, 0),
        datetime(2021, 10, 9, 0, 0)
    ]}, 'col1')
    check_obj = DateTrainTestLeakageDuplicates()
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.31, 0.01))


def test_limit_dates_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 5, 0, 0)
    ]}, 'col1')
    val_ds = dataset_from_dict({'col1': [
        datetime(2021, 9, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 6, 0, 0),

    ]}, 'col1')
    check_obj = DateTrainTestLeakageDuplicates(n_to_show=1)
    result = check_obj.run(train_ds, val_ds)
    assert_that(result.value, close_to(0.5, 0.01))
    assert_that(result.display, has_length(greater_than(0)))


def test_limit_dates_from_val_in_train_without_display():
    train_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 5, 0, 0)
    ]}, 'col1')
    val_ds = dataset_from_dict({'col1': [
        datetime(2021, 9, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 6, 0, 0),

    ]}, 'col1')
    check_obj = DateTrainTestLeakageDuplicates(n_to_show=1)
    result = check_obj.run(train_ds, val_ds, with_display=False)
    assert_that(result.value, close_to(0.5, 0.01))
    assert_that(result.display, has_length(0))


def test_no_dates_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
    ]}, 'col1')
    val_ds = dataset_from_dict({'col1': [
        datetime(2021, 9, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 6, 0, 0),

    ]}, 'col1')
    check_obj = DateTrainTestLeakageDuplicates()
    assert_that(check_obj.run(train_ds, val_ds).value, equal_to(0))


def test_dataset_wrong_input():
    x = 'wrong_input'
    assert_that(
        calling(DateTrainTestLeakageDuplicates().run).with_args(x, x),
        raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str'))
    assert_that(
        calling(DateTrainTestLeakageOverlap().run).with_args(x, x),
        raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str'))


def test_dataset_no_index():
    ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]})
    assert_that(
        calling(DateTrainTestLeakageDuplicates().run).with_args(ds, ds),
        raises(DatasetValidationError,
               'Dataset does not contain a datetime'))
    assert_that(
        calling(DateTrainTestLeakageOverlap().run).with_args(ds, ds),
        raises(DatasetValidationError,
               'Dataset does not contain a datetime'))


def test_dates_from_val_before_train():
    train_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 5, 0, 0)
    ]}, 'col1')
    val_ds = dataset_from_dict({'col1': [
        datetime(2021, 9, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 9, 0, 0),
        datetime(2021, 10, 9, 0, 0)
    ]}, 'col1')
    check_obj = DateTrainTestLeakageOverlap()
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.18, 0.01))


def test_no_dates_from_val_before_train():
    train_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 5, 0, 0)
    ]}, 'col1')
    val_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),

    ]}, 'col1')
    check_obj = DateTrainTestLeakageOverlap()
    assert_that(check_obj.run(train_ds, val_ds).value, equal_to(0))


def test_no_dates_from_val_before_train_date_in_index():
    train_ds = Dataset(pd.DataFrame(np.ones((7, 1)), columns=['col1'], index=[
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 5, 0, 0)
    ]), set_datetime_from_dataframe_index=True)
    val_ds = Dataset(pd.DataFrame(np.ones((3, 1)), columns=['col1'], index=[
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),
    ]), set_datetime_from_dataframe_index=True)
    check_obj = DateTrainTestLeakageOverlap()
    assert_that(check_obj.run(train_ds, val_ds).value, equal_to(0))


def test_nan():
    train_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        np.nan
    ]}, 'col1')
    val_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        np.nan

    ]}, 'col1')
    check_obj = DateTrainTestLeakageOverlap()
    assert_that(check_obj.run(train_ds, val_ds).value, equal_to(0))


def test_condition_fail_on_overlap():
    # Arrange
    train_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 5, 0, 0)
    ]}, 'col1')
    val_ds = dataset_from_dict({'col1': [
        datetime(2021, 9, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 9, 0, 0),
        datetime(2021, 10, 9, 0, 0)
    ]}, 'col1')

    check = DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal(0.17)

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Date leakage ratio is less or equal to 17%',
                               details='Found 18.18% leaked dates')
    ))


def test_condition_fail_on_overlap_date_in_index():
    # Arrange
    train_ds = Dataset(pd.DataFrame(np.ones((14, 1)), columns=['col1'], index=[
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 5, 0, 0)
    ]), set_datetime_from_dataframe_index=True)

    val_ds = Dataset(pd.DataFrame(np.ones((11, 1)), columns=['col1'], index=[
        datetime(2021, 9, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 9, 0, 0),
        datetime(2021, 10, 9, 0, 0)
    ]), set_datetime_from_dataframe_index=True)

    check = DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal(0.17)

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Date leakage ratio is less or equal to 17%',
                               details='Found 18.18% leaked dates')
    ))


def test_condition_on_overlap():
    # Arrange
    train_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),

    ]}, 'col1')
    val_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 9, 0, 0),
        datetime(2021, 10, 9, 0, 0)
    ]}, 'col1')

    check = DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal()

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Date leakage ratio is less or equal to 0%',
                               details='No leaked dates found')
    ))


def test_condition_fail_on_duplicates():
    # Arrange
    train_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 5, 0, 0)
    ]}, 'col1')
    val_ds = dataset_from_dict({'col1': [
        datetime(2021, 9, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 9, 0, 0),
        datetime(2021, 10, 9, 0, 0)
    ]}, 'col1')

    check = DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_less_or_equal(0.1)

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Date leakage ratio is less or equal to 10%',
                               details='Found 18.18% leaked dates')
    ))


def test_condition_pass_on_duplicates():
    # Arrange
    train_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 1, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 2, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 3, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),
        datetime(2021, 10, 4, 0, 0),

    ]}, 'col1')
    val_ds = dataset_from_dict({'col1': [
        datetime(2021, 10, 5, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 6, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 7, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 8, 0, 0),
        datetime(2021, 10, 9, 0, 0),
        datetime(2021, 10, 9, 0, 0)
    ]}, 'col1')

    check = DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_less_or_equal()

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Date leakage ratio is less or equal to 0%',
                               details='No leaked dates found')
    ))