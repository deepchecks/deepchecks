"""
Contains unit tests for the single_feature_contribution check
"""
from datetime import datetime

import pandas as pd

from hamcrest import assert_that, close_to, calling, raises, equal_to

from mlchecks import Dataset
from mlchecks.checks.leakage import DateTrainValidationLeakageOverlap, DateTrainValidationLeakageDuplicates
from mlchecks.utils import MLChecksValueError


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
    check_obj = DateTrainValidationLeakageDuplicates()
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
    check_obj = DateTrainValidationLeakageDuplicates(n_to_show=1)
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
    check_obj = DateTrainValidationLeakageDuplicates()
    assert_that(check_obj.run(train_ds, val_ds).value, equal_to(0))


def test_dataset_wrong_input():
    x = 'wrong_input'
    assert_that(
        calling(DateTrainValidationLeakageDuplicates().run).with_args(x, x),
        raises(MLChecksValueError, 'Check DateTrainValidationLeakageDuplicates '
                                   'requires dataset to be of type Dataset. instead got: str'))
    assert_that(
        calling(DateTrainValidationLeakageOverlap().run).with_args(x, x),
        raises(MLChecksValueError, 'Check DateTrainValidationLeakageOverlap '
                                   'requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_index():
    ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]})
    assert_that(
        calling(DateTrainValidationLeakageDuplicates().run).with_args(ds, ds),
        raises(MLChecksValueError, 'Check DateTrainValidationLeakageDuplicates '
                                   'requires dataset to have a date column'))
    assert_that(
        calling(DateTrainValidationLeakageOverlap().run).with_args(ds, ds),
        raises(MLChecksValueError, 'Check DateTrainValidationLeakageOverlap '
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
    check_obj = DateTrainValidationLeakageOverlap()
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
    check_obj = DateTrainValidationLeakageOverlap()
    assert_that(check_obj.run(train_ds, val_ds).value, equal_to(0))
