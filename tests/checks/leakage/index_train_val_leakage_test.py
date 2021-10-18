"""
Contains unit tests for the single_feature_contribution check
"""
import pandas as pd

from hamcrest import assert_that, close_to, calling, raises

from mlchecks import Dataset
from mlchecks.checks.leakage.index_leakage import index_train_validation_leakage, IndexTrainValidationLeakage
from mlchecks.utils import MLChecksValueError


def dataset_from_dict(d: dict, index_name: str = None) -> Dataset:
    dataframe = pd.DataFrame(data=d)
    return Dataset(dataframe, index=index_name)


def test_indexes_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 5, 6, 7]}, 'col1')
    assert_that(index_train_validation_leakage(train_ds, val_ds).value, close_to(0.25, 0.01))
    check_obj = IndexTrainValidationLeakage()
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.25, 0.01))


def test_limit_indexes_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 3, 6, 7]}, 'col1')
    assert_that(index_train_validation_leakage(train_ds, val_ds, n_index_to_show=1).value, close_to(0.5, 0.01))
    check_obj = IndexTrainValidationLeakage(n_index_to_show=1)
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.5, 0.01))


def test_no_indexes_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [20, 5, 6, 7]}, 'col1')
    assert_that(index_train_validation_leakage(train_ds, val_ds).value, close_to(0, 0.01))
    check_obj = IndexTrainValidationLeakage()
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0, 0.01))


def test_dataset_wrong_input():
    x = 'wrong_input'
    assert_that(
        calling(index_train_validation_leakage).with_args(x, x),
        raises(MLChecksValueError, 'function index_train_validation_leakage requires dataset to be of type Dataset. '
                                   'instead got: str'))


def test_dataset_no_index():
    ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]})
    assert_that(
        calling(index_train_validation_leakage).with_args(ds, ds),
        raises(MLChecksValueError, 'function index_train_validation_leakage requires dataset to have an index column'))
