# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""Contains unit tests for the columns_info check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, equal_to, calling, raises, has_length

from deepchecks.base import Dataset
from deepchecks.checks.overview import ColumnsInfo
from deepchecks.errors import DeepchecksValueError


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    assert_that(calling(ColumnsInfo().run).with_args(x),
                raises(DeepchecksValueError, 'dataset must be of type DataFrame or Dataset. instead got: str'))


def test_columns_info():
    num_fe = np.random.rand(600)
    cat_fe = np.random.randint(5, size=600)
    date = range(1635693229, 1635693829)
    index = range(600)
    data = {'index': index, 'date': date, 'a': cat_fe, 'b': num_fe, 'c': num_fe, 'label': cat_fe}
    df = pd.DataFrame.from_dict(data)

    dataset = Dataset(df, label='label', date='date', index='index', features=['a', 'b'])
    # Arrange
    check = ColumnsInfo()
    # Act
    result_ds, result_df = check.run(dataset).value, check.run(df).value
    # Assert
    expected_res_ds = {'index': 'index', 'date': 'date', 'a': 'categorical feature',
                    'b': 'numerical feature', 'c': 'other', 'label': 'label'}
    assert_that(result_ds, equal_to(expected_res_ds))

    # in df all columns are features
    expected_res_df = expected_res_ds
    expected_res_df['index'] = 'numerical feature'
    expected_res_df['date'] = 'numerical feature'
    expected_res_df['label'] = 'categorical feature'
    expected_res_df['c'] = 'numerical feature'
    assert_that(result_df, equal_to(expected_res_df))

def test_fi_n_top(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    # Arrange
    check = ColumnsInfo(n_top_columns=3)
    # Act
    result_ds = check.run(train, clf).value
    # Assert
    assert_that(result_ds, has_length(3))
