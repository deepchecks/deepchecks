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
"""Contains unit tests for the identifier_label_correlation check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, calling, close_to, greater_than, has_items, has_length, is_in, raises

from deepchecks.core.errors import DatasetValidationError, DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.tabular.checks.data_integrity.identifier_label_correlation import IdentifierLabelCorrelation
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def generate_dataframe_and_expected():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
    df['label'] = df['x2'] + 0.1 * df['x1']

    return df, {'x2': 0.42, 'x1': 0.0, 'x3': 0.0}


def test_assert_identifier_label_correlation():
    df, expected = generate_dataframe_and_expected()
    result = IdentifierLabelCorrelation().run(dataset=Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                              cat_features=[]))
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))
    assert_that(result.display, has_length(greater_than(0)))


def test_assert_identifier_label_correlation_without_display():
    df, expected = generate_dataframe_and_expected()
    result = IdentifierLabelCorrelation().run(dataset=Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                              cat_features=[]), with_display=False)
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))
    assert_that(result.display, has_length(0))


def test_identifier_label_correlation_with_extracted_from_dataframe_index():
    df, expected = generate_dataframe_and_expected()
    df.set_index('x3', inplace=True)
    dataset = Dataset(df=df, label='label', set_index_from_dataframe_index=True, cat_features=[])
    result = IdentifierLabelCorrelation().run(dataset=dataset)
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_identifier_label_correlation_with_extracted_from_dataframe_datatime_index():
    df, expected = generate_dataframe_and_expected()
    df.set_index('x2', inplace=True)
    dataset = Dataset(df=df, label='label', set_datetime_from_dataframe_index=True, cat_features=[])
    result = IdentifierLabelCorrelation().run(dataset=dataset)
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_dataset_wrong_input():
    wrong = 'wrong_input'
    assert_that(
        calling(IdentifierLabelCorrelation().run).with_args(wrong),
        raises(
            DeepchecksValueError,
            'non-empty instance of Dataset or DataFrame was expected, instead got str')
    )


def test_dataset_no_label():
    df, _ = generate_dataframe_and_expected()
    df = Dataset(df)
    assert_that(
        calling(IdentifierLabelCorrelation().run).with_args(dataset=df),
        raises(DeepchecksNotSupportedError,
               'Dataset does not contain a label column')
    )


def test_dataset_only_label():
    df, _ = generate_dataframe_and_expected()
    df = Dataset(df, label='label')
    assert_that(
        calling(IdentifierLabelCorrelation().run).with_args(dataset=df),
        raises(
            DatasetValidationError,
            'Dataset does not contain an index or a datetime')
    )


def test_assert_label_correlation_class():
    df, expected = generate_dataframe_and_expected()
    identifier_leakage_check = IdentifierLabelCorrelation()
    result = identifier_leakage_check.run(dataset=Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                          cat_features=[]))
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_nan():
    df, expected = generate_dataframe_and_expected()
    nan_df = df.append(pd.DataFrame({'x1': [np.nan],
                                     'x2': [np.nan],
                                     'x3': [np.nan],
                                     'label': [0]}))

    result = IdentifierLabelCorrelation().run(dataset=Dataset(nan_df, label='label', datetime_name='x2',
                                                              index_name='x3', cat_features=[]))
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_condition_pps_pass():
    df, expected = generate_dataframe_and_expected()

    check = IdentifierLabelCorrelation().add_condition_pps_less_or_equal(0.5)

    # Act
    result = check.conditions_decision(check.run(Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                         cat_features=[])))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Passed for 2 relevant columns',
                               name='Identifier columns PPS is less or equal to 0.5')
    ))


def test_condition_pps_fail():
    df, expected = generate_dataframe_and_expected()

    check = IdentifierLabelCorrelation(n_samples=None).add_condition_pps_less_or_equal(0.2)

    # Act
    result = check.conditions_decision(check.run(Dataset(df, label='label', datetime_name='x2', index_name='x3',
                                                         cat_features=[])))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 1 out of 2 columns with PPS above threshold: {\'x2\': \'0.42\'}',
                               name='Identifier columns PPS is less or equal to 0.2')
    ))