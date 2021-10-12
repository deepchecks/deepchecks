"""
Contains unit tests for the single_feature_contribution check
"""
import numpy as np
import pandas as pd

from mlchecks import Dataset
from mlchecks.checks.leakage.single_feature_contribution import single_feature_contribution
from hamcrest import *
from mlchecks.utils import MLChecksValueError

def util_generate_dataframe_and_expected():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
    df['x4'] = df['x1'] * 0.5 + df['x2']
    df['label'] = df['x2'] + 0.1 * df['x1']
    df['x5'] = df['label'].apply(lambda x: 'v1' if x < 0 else 'v2')

    return df, {'x2': 0.8410436710134066,
                'x4': 0.5196251242216743,
                'x5': 0.43077684482083267,
                'x1': 0.0,
                'x3': 0.0}


def test_assert_single_feature_contribution():

    df, expected = util_generate_dataframe_and_expected()
    result = single_feature_contribution(dataset=Dataset(df, label='label'))

    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_dataset_wrong_input():
    X = "wrong_input"
    assert_that(calling(single_feature_contribution).with_args(X),
                raises(MLChecksValueError, 'single_feature_contribution check must receive a Dataset object'))


def test_dataset_no_label():
    df, _ = util_generate_dataframe_and_expected()
    assert_that(calling(single_feature_contribution).with_args(dataset=Dataset(df)),
                raises(MLChecksValueError, 'single_feature_contribution requires dataset to have a label'))
