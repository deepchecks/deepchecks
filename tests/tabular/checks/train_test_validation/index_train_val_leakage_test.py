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
Contains unit tests for the index leakage check
"""
import numpy as np
import pandas as pd
from hamcrest import assert_that, calling, close_to, greater_than, has_items, has_length, raises

from deepchecks.core.errors import DatasetValidationError, DeepchecksValueError
from deepchecks.tabular.checks.train_test_validation.index_leakage import IndexTrainTestLeakage
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def dataset_from_dict(d: dict, index_name: str = None) -> Dataset:
    dataframe = pd.DataFrame(data=d)
    return Dataset(dataframe, index_name=index_name)


def test_indexes_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 5, 6, 7]}, 'col1')
    check_obj = IndexTrainTestLeakage()
    result = check_obj.run(train_ds, val_ds)
    assert_that(result.value, close_to(0.25, 0.01))
    assert_that(result.display, has_length(greater_than(0)))


def test_indexes_from_val_in_train_without_display():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 5, 6, 7]}, 'col1')
    check_obj = IndexTrainTestLeakage()
    result = check_obj.run(train_ds, val_ds, with_display=False)
    assert_that(result.value, close_to(0.25, 0.01))
    assert_that(result.display, has_length(0))


def test_limit_indexes_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 3, 6, 7]}, 'col1')
    check_obj = IndexTrainTestLeakage(n_to_show=1)
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.5, 0.01))


def test_no_indexes_from_val_in_train():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [20, 5, 6, 7]}, 'col1')
    check_obj = IndexTrainTestLeakage()
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0, 0.01))


def test_dataset_wrong_input():
    x = 'wrong_input'
    assert_that(
        calling(IndexTrainTestLeakage().run).with_args(x, x),
        raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str'))


def test_dataset_no_index():
    ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]})
    assert_that(
        calling(IndexTrainTestLeakage().run).with_args(ds, ds),
        raises(DatasetValidationError, 'Dataset does not contain an index'))


def test_nan():
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11, np.nan]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 5, 6, 7, np.nan]}, 'col1')
    check_obj = IndexTrainTestLeakage(n_to_show=1)
    assert_that(check_obj.run(train_ds, val_ds).value, close_to(0.2, 0.01))


def test_condition_leakage_fail():
    # Arrange
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11, np.nan]}, 'col1')
    val_ds = dataset_from_dict({'col1': [4, 5, 6, 7, np.nan]}, 'col1')
    check = IndexTrainTestLeakage(n_to_show=1).add_condition_ratio_less_or_equal(max_ratio=0.19)

    result = check.conditions_decision(check.run(train_ds, val_ds))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 20% of index leakage',
                               name='Ratio of leaking indices is less or equal to 19%')
    ))


def test_condition_leakage_passesl():
    # Arrange
    train_ds = dataset_from_dict({'col1': [1, 2, 3, 4, 10, 11]}, 'col1')
    val_ds = dataset_from_dict({'col1': [20, 5, 6, 7]}, 'col1')
    check = IndexTrainTestLeakage(n_to_show=1).add_condition_ratio_less_or_equal()

    result = check.conditions_decision(check.run(train_ds, val_ds))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='No index leakage found',
                               name='Ratio of leaking indices is less or equal to 0%')
    ))
