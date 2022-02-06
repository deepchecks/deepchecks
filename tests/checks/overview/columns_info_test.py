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
"""Contains unit tests for the columns_info check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, equal_to, calling, raises, has_length

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks.overview import ColumnsInfo
from deepchecks.core.errors import DeepchecksValueError


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(ColumnsInfo().run).with_args(x),
        raises(
            DeepchecksValueError,
            'non-empty instance of Dataset or DataFrame was expected, instead got str'
        )
    )


def test_columns_info():
    num_fe = np.random.rand(600)
    cat_fe = np.random.randint(5, size=600)
    date = range(1635693229, 1635693829)
    index = range(600)
    data = {'index': index, 'date': date, 'a': cat_fe, 'b': num_fe, 'c': num_fe, 'label': cat_fe}
    df = pd.DataFrame.from_dict(data)

    dataset = Dataset(df, label='label', datetime_name='date', index_name='index', features=['a', 'b'])
    # Arrange
    check = ColumnsInfo()
    # Act
    result_ds, result_df = check.run(dataset).value, check.run(df).value
    # Assert
    expected_res_ds = {'index': 'index', 'date': 'date', 'a': 'categorical feature',
                       'b': 'numerical feature', 'c': 'other', 'label': 'label'}
    assert_that(result_ds, equal_to(expected_res_ds))

    # in df all columns are other
    expected_res_df = {'index': 'other', 'date': 'other', 'a': 'other',
                       'b': 'other', 'c': 'other', 'label': 'other'}
    assert_that(result_df, equal_to(expected_res_df))


def test_fi_n_top(diabetes_split_dataset_and_model):
    train, _, clf = diabetes_split_dataset_and_model
    # Arrange
    check = ColumnsInfo(n_top_columns=3)
    # Act
    result_ds = check.run(train, clf).value
    # Assert
    assert_that(result_ds, has_length(3))
