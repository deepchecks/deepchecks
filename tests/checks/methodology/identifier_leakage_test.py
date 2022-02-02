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
"""Contains unit tests for the identifier_leakage check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, is_in, close_to, calling, raises, has_items

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks.methodology.identifier_leakage import IdentifierLeakage
from deepchecks.core.errors import DeepchecksValueError, DatasetValidationError, DeepchecksNotSupportedError

from tests.checks.utils import equal_condition_result


def util_generate_dataframe_and_expected():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
    df['label'] = df['x2'] + 0.1 * df['x1']

    return df, {'x2': 0.42, 'x1': 0.0, 'x3': 0.0}


def test_assert_identifier_leakage():
    df, expected = util_generate_dataframe_and_expected()
    result = IdentifierLeakage().run(dataset=Dataset(df, label='label', datetime_name='x2', index_name='x3'))
    print(result.value)
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_dataset_wrong_input():
    wrong = 'wrong_input'
    assert_that(
        calling(IdentifierLeakage().run).with_args(wrong),
        raises(
            DeepchecksValueError,
            'non-empty instance of Dataset or DataFrame was expected, instead got str')
    )


def test_dataset_no_label():
    df, _ = util_generate_dataframe_and_expected()
    df = Dataset(df)
    assert_that(
        calling(IdentifierLeakage().run).with_args(dataset=df),
        raises(DeepchecksNotSupportedError,
               'There is no label defined to use. Did you pass a DataFrame instead of a Dataset?')
    )


def test_dataset_only_label():
    df, _ = util_generate_dataframe_and_expected()
    df = Dataset(df, label='label')
    assert_that(
        calling(IdentifierLeakage().run).with_args(dataset=df),
        raises(
            DatasetValidationError,
            'Check is irrelevant for Datasets without index or date column')
    )


def test_assert_identifier_leakage_class():
    df, expected = util_generate_dataframe_and_expected()
    identifier_leakage_check = IdentifierLeakage()
    result = identifier_leakage_check.run(dataset=Dataset(df, label='label', datetime_name='x2', index_name='x3'))
    print(result.value)
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_nan():
    df, expected = util_generate_dataframe_and_expected()
    nan_df = df.append(pd.DataFrame({'x1': [np.nan],
                                     'x2': [np.nan],
                                     'x3': [np.nan],
                                     'label': [0]}))

    result = IdentifierLeakage().run(dataset=Dataset(nan_df, label='label', datetime_name='x2', index_name='x3'))
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_condition_pps_pass():
    df, expected = util_generate_dataframe_and_expected()

    check = IdentifierLeakage().add_condition_pps_not_greater_than(0.5)

    # Act
    result = check.conditions_decision(check.run(Dataset(df, label='label', datetime_name='x2', index_name='x3')))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Identifier columns PPS is not greater than 0.5')
    ))


def test_condition_pps_fail():
    df, expected = util_generate_dataframe_and_expected()

    check = IdentifierLeakage().add_condition_pps_not_greater_than(0.2)

    # Act
    result = check.conditions_decision(check.run(Dataset(df, label='label', datetime_name='x2', index_name='x3')))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found columns with PPS above threshold: {\'x2\': \'0.42\'}',
                               name='Identifier columns PPS is not greater than 0.2')
    ))