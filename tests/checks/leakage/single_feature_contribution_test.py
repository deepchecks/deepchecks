"""
Contains unit tests for the single_feature_contribution check
"""
import numpy as np
import pandas as pd

from mlchecks import Dataset
from mlchecks.checks.leakage.single_feature_contribution import single_feature_contribution, \
    single_feature_contribution_train_validation
from hamcrest import *
from mlchecks.utils import MLChecksValueError


def util_generate_dataframe_and_expected():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
    df['x4'] = df['x1'] * 0.5 + df['x2']
    df['label'] = df['x2'] + 0.1 * df['x1']
    df['x5'] = df['label'].apply(lambda x: 'v1' if x < 0 else 'v2')

    return df, {'x2': 0.84, 'x4': 0.53, 'x5': 0.42, 'x1': 0.0, 'x3': 0.0}


def util_generate_second_similar_dataframe_and_expected():
    np.random.seed(42)
    df, _ = util_generate_dataframe_and_expected()
    df2 = df.copy()
    df2['x2'] = df['x2'] + 0.5 * df['x1']
    df2['x3'] = 0.3 * df['x3'] + df['label']

    return df, df2, {'x1': 0.0, 'x2': -0.3, 'x3': 0.5, 'x4': 0.0, 'x5': 0.0}


def test_assert_single_feature_contribution():
    df, expected = util_generate_dataframe_and_expected()
    result = single_feature_contribution(dataset=Dataset(df, label='label'))
    print(result.value)
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_dataset_wrong_input():
    X = "wrong_input"
    assert_that(
        calling(single_feature_contribution).with_args(X),
        raises(MLChecksValueError, 'function single_feature_contribution requires dataset to be of type Dataset. '
                                   'instead got: str'))


def test_dataset_no_label():
    df, _ = util_generate_dataframe_and_expected()
    df = Dataset(df)
    assert_that(
        calling(single_feature_contribution).with_args(dataset=df),
        raises(MLChecksValueError, 'function single_feature_contribution requires dataset to have a label column'))


def test_trainval_assert_single_feature_contribution():
    df, df2, expected = util_generate_second_similar_dataframe_and_expected()
    result = single_feature_contribution_train_validation(train_dataset=Dataset(df, label='label'),
                                                          validation_dataset=Dataset(df2, label='label'))
    print(result.value)
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_trainval_dataset_wrong_input():
    X = "wrong_input"
    assert_that(
        calling(single_feature_contribution_train_validation).with_args(X, X),
        raises(MLChecksValueError, 'function single_feature_contribution requires dataset to be of type Dataset. '
                                   'instead got: str'))


def test_trainval_dataset_no_label():
    df, df2, expected = util_generate_second_similar_dataframe_and_expected()
    assert_that(
        calling(single_feature_contribution_train_validation).with_args(train_dataset=Dataset(df),
                                                                        validation_dataset=Dataset(df2)),
        raises(MLChecksValueError, 'function single_feature_contribution requires dataset to have a label column'))
