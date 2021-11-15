"""
Contains unit tests for the single_feature_contribution check
"""
import numpy as np
import pandas as pd

from hamcrest import assert_that, close_to, calling, raises

from deepchecks import Dataset
from deepchecks.checks.leakage.index_leakage import IndexTrainValidationLeakage
from deepchecks.utils import DeepchecksValueError


def dataset_from_dict(d: dict, index_name: str = None) -> Dataset:
    dataframe = pd.DataFrame(data=d)
    return Dataset(dataframe, index=index_name)


def test_indexes_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 5, 6, 7]}, 'col1')
    check_obj = IndexTrainValidationLeakage()
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.25, 0.01))


def test_limit_indexes_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 3, 6, 7]}, 'col1')
    check_obj = IndexTrainValidationLeakage(n_index_to_show=1)
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.5, 0.01))


def test_no_indexes_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [20, 5, 6, 7]}, 'col1')
    check_obj = IndexTrainValidationLeakage()
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0, 0.01))


def test_dataset_wrong_input():
    x = 'wrong_input'
    assert_that(
        calling(IndexTrainValidationLeakage().run).with_args(x, x),
        raises(DeepchecksValueError, 'Check IndexTrainValidationLeakage requires dataset to be of type Dataset. '
                                   'instead got: str'))


def test_dataset_no_index():
    ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]})
    assert_that(
        calling(IndexTrainValidationLeakage().run).with_args(ds, ds),
        raises(DeepchecksValueError, 'Check IndexTrainValidationLeakage requires dataset to have an index column'))


def test_nan():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11, np.nan]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 5, 6, 7, np.nan]}, 'col1')
    check_obj = IndexTrainValidationLeakage(n_index_to_show=1)
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.2, 0.01))
