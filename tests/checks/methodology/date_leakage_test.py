"""
Contains unit tests for the single_feature_contribution check
"""
from datetime import datetime

import numpy as np
import pandas as pd

from hamcrest import assert_that, close_to, calling, raises, equal_to, has_items

from deepchecks import Dataset
from deepchecks.checks.methodology import DateTrainTestLeakageOverlap, DateTrainTestLeakageDuplicates
from deepchecks.errors import DeepchecksValueError
from tests.checks.utils import equal_condition_result


def dataset_from_dict(d: dict, date_name: str = None) -> Dataset:
    dataframe = pd.DataFrame(data=d)
    return Dataset(dataframe, date=date_name)


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
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.182, 0.01))


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
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.5, 0.01))


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
        raises(DeepchecksValueError, 'Check DateTrainTestLeakageDuplicates '
                                   'requires dataset to be of type Dataset. instead got: str'))
    assert_that(
        calling(DateTrainTestLeakageOverlap().run).with_args(x, x),
        raises(DeepchecksValueError, 'Check DateTrainTestLeakageOverlap '
                                   'requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_index():
    ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]})
    assert_that(
        calling(DateTrainTestLeakageDuplicates().run).with_args(ds, ds),
        raises(DeepchecksValueError, 'Check DateTrainTestLeakageDuplicates '
                                   'requires dataset to have a date column'))
    assert_that(
        calling(DateTrainTestLeakageOverlap().run).with_args(ds, ds),
        raises(DeepchecksValueError, 'Check DateTrainTestLeakageOverlap '
                                   'requires dataset to have a date column'))

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
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.27, 0.01))


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

    check = DateTrainTestLeakageOverlap().add_condition_leakage_ratio_not_greater_than(0.2)

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Date leakage ratio is not greater than 20.00%',
                               details='Found 27.27% leaked dates')
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

    check = DateTrainTestLeakageOverlap().add_condition_leakage_ratio_not_greater_than()

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Date leakage ratio is not greater than 0%',
                               details='')
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

    check = DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_not_greater_than(0.1)

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Date leakage ratio is not greater than 10.00%',
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

    check = DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_not_greater_than()

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Date leakage ratio is not greater than 0%',
                               details='')
    ))