"""Contains unit tests for the columns_info check."""
from mlchecks.base import Dataset
from mlchecks.checks.overview import ColumnsInfo
from mlchecks.utils import MLChecksValueError

import numpy as np
import pandas as pd
from hamcrest import assert_that, equal_to, calling, raises


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    assert_that(calling(ColumnsInfo().run).with_args(x),
                raises(MLChecksValueError, 'dataset must be of type DataFrame or Dataset. instead got: str'))


def test_columns_info():
    num_fe = np.random.rand(200)
    cat_fe = np.random.randint(10, size=200)
    date = range(1635693229, 1635693429)
    index = range(200)
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
