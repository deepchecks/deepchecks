"""Contains unit tests for the identifier_leakage check."""
import numpy as np
import pandas as pd

from mlchecks import Dataset
from mlchecks.checks.leakage.identifier_leakage import IdentifierLeakage
from mlchecks.utils import MLChecksValueError

from hamcrest import assert_that, is_in, close_to, calling, raises


def util_generate_dataframe_and_expected():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
    df['label'] = df['x2'] + 0.1 * df['x1']

    return df, {'x2': 0.42, 'x1': 0.0, 'x3': 0.0}


def test_assert_identifier_leakage():
    df, expected = util_generate_dataframe_and_expected()
    result = IdentifierLeakage().run(dataset=Dataset(df, label='label', date='x2', index='x3'))
    print(result.value)
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_dataset_wrong_input():
    wrong = 'wrong_input'
    assert_that(
        calling(IdentifierLeakage().run).with_args(wrong),
        raises(MLChecksValueError, 'function _identifier_leakage requires dataset to be of type Dataset. '
                                   'instead got: str'))


def test_dataset_no_label():
    df, _ = util_generate_dataframe_and_expected()
    df = Dataset(df)
    assert_that(
        calling(IdentifierLeakage().run).with_args(dataset=df),
        raises(MLChecksValueError, 'function _identifier_leakage requires dataset to have a label column'))


def test_dataset_only_label():
    df, _ = util_generate_dataframe_and_expected()
    df = Dataset(df, label='label')
    assert_that(
        calling(IdentifierLeakage().run).with_args(dataset=df),
        raises(MLChecksValueError, 'Dataset needs to have a date or index column'))


def test_assert_identifier_leakage_class():
    df, expected = util_generate_dataframe_and_expected()
    identifier_leakage_check = IdentifierLeakage()
    result = identifier_leakage_check.run(dataset=Dataset(df, label='label', date='x2', index='x3'))
    print(result.value)
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))
