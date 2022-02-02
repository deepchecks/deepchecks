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
"""Contains unit tests for the single_feature_contribution check."""
import numpy as np
import pandas as pd
from hamcrest import assert_that, close_to, calling, raises, has_entries, has_length

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks.methodology import SingleFeatureContribution, SingleFeatureContributionTrainTest
from deepchecks.core.errors import DeepchecksValueError, DatasetValidationError, DeepchecksNotSupportedError

from tests.checks.utils import equal_condition_result


def util_generate_dataframe_and_expected():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
    df['x4'] = df['x1'] * 0.5 + df['x2']
    df['label'] = df['x2'] + 0.1 * df['x1']
    df['x5'] = df['label'].apply(lambda x: 'v1' if x < 0 else 'v2')

    return df, {'x2': close_to(0.84, 0.01), 'x4': close_to(0.53, 0.01), 'x5': close_to(0.42, 0.01),
                'x1': close_to(0, 0.01), 'x3': close_to(0, 0.01)}


def util_generate_second_similar_dataframe_and_expected():
    np.random.seed(42)
    df, _ = util_generate_dataframe_and_expected()
    df2 = df.copy()
    df2['x2'] = df['x2'] + 0.5 * df['x1']
    df2['x3'] = 0.3 * df['x3'] + df['label']

    return df, df2, {'x1': close_to(0, 0.1), 'x2': close_to(0.3, 0.1), 'x3': close_to(-0.54, 0.1),
                     'x4': close_to(0, 0.1), 'x5': close_to(0, 0.1)}


def test_assert_single_feature_contribution():
    df, expected = util_generate_dataframe_and_expected()
    result = SingleFeatureContribution().run(dataset=Dataset(df, label='label'))

    assert_that(result.value, has_entries(expected))


def test_show_top_single_feature_contribution():
    # Arrange
    df, expected = util_generate_dataframe_and_expected()
    check = SingleFeatureContribution(n_show_top=3)

    # Act
    result = check.run(dataset=Dataset(df, label='label'))

    # Assert
    assert_that(result.value, has_length(5))
    assert_that(result.value, has_entries(expected))


def test_dataset_wrong_input():
    wrong = 'wrong_input'
    assert_that(
        calling(SingleFeatureContribution().run).with_args(wrong),
        raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str'))


def test_dataset_no_label():
    df, _ = util_generate_dataframe_and_expected()
    df = Dataset(df)
    assert_that(
        calling(SingleFeatureContribution().run).with_args(dataset=df),
        raises(DeepchecksNotSupportedError,
               'There is no label defined to use. Did you pass a DataFrame instead of a Dataset?'))


def test_trainval_assert_single_feature_contribution():
    df, df2, expected = util_generate_second_similar_dataframe_and_expected()
    result = SingleFeatureContributionTrainTest().run(train_dataset=Dataset(df, label='label'),
                                                      test_dataset=Dataset(df2, label='label'))

    assert_that(result.value['train-test difference'], has_entries(expected))


def test_trainval_show_top_single_feature_contribution():
    df, df2, expected = util_generate_second_similar_dataframe_and_expected()
    result = SingleFeatureContributionTrainTest(n_show_top=3).run(train_dataset=Dataset(df, label='label'),
                                                                  test_dataset=Dataset(df2, label='label'))
    assert_that(result.value['train-test difference'], has_length(5))
    assert_that(result.value['train-test difference'], has_entries(expected))


def test_trainval_dataset_wrong_input():
    wrong = 'wrong_input'
    assert_that(
        calling(SingleFeatureContributionTrainTest().run).with_args(wrong, wrong),
        raises(
            DeepchecksValueError,
            'non-empty instance of Dataset or DataFrame was expected, instead got str')
    )


def test_trainval_dataset_no_label():
    df, df2, _ = util_generate_second_similar_dataframe_and_expected()
    assert_that(
        calling(SingleFeatureContributionTrainTest().run).with_args(
            train_dataset=Dataset(df),
            test_dataset=Dataset(df2)),
        raises(
            DeepchecksNotSupportedError,
            'There is no label defined to use. Did you pass a DataFrame instead of a Dataset?')
    )


def test_trainval_dataset_diff_columns():
    df, df2, _ = util_generate_second_similar_dataframe_and_expected()
    df = df.rename({'x2': 'x6'}, axis=1)
    assert_that(
        calling(SingleFeatureContributionTrainTest().run).with_args(
            train_dataset=Dataset(df, label='label'),
            test_dataset=Dataset(df2, label='label')),
        raises(
            DatasetValidationError,
            'train and test requires to share the same features columns')
    )


def test_all_features_pps_upper_bound_condition_that_should_not_pass():
    # Arrange
    df, _ = util_generate_dataframe_and_expected()
    dataset = Dataset(df, label="label")
    condition_value = 0.4
    check = SingleFeatureContribution().add_condition_feature_pps_not_greater_than(condition_value)

    # Act
    condition_result, *_ = check.conditions_decision(check.run(dataset))

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Features\' Predictive Power Score is not greater than {condition_value}',
        details='Features with PPS above threshold: {\'x2\': \'0.84\', \'x4\': \'0.53\', \'x5\': \'0.42\'}'
    ))


def test_all_features_pps_upper_bound_condition_that_should_pass():
    # Arrange
    df, expected = util_generate_dataframe_and_expected()
    dataset = Dataset(df, label="label")
    condition_value = 0.9
    check = SingleFeatureContribution().add_condition_feature_pps_not_greater_than(condition_value)

    # Act
    condition_result, *_ = check.conditions_decision(check.run(dataset))

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        name=f'Features\' Predictive Power Score is not greater than {condition_value}',
    ))


def test_train_test_condition_pps_difference_pass():
    # Arrange
    df, df2, expected = util_generate_second_similar_dataframe_and_expected()
    condition_value = 0.4
    check = SingleFeatureContributionTrainTest().add_condition_feature_pps_difference_not_greater_than(condition_value)

    # Act
    result = SingleFeatureContributionTrainTest().run(train_dataset=Dataset(df, label='label'),
                                                      test_dataset=Dataset(df2, label='label'))
    condition_result, *_ = check.conditions_decision(result)
    print(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        name=f'Train-Test features\' Predictive Power Score difference is not greater than {condition_value}'
    ))


def test_train_test_condition_pps_difference_fail():
    # Arrange
    df, df2, expected = util_generate_second_similar_dataframe_and_expected()
    condition_value = 0.01
    check = SingleFeatureContributionTrainTest().add_condition_feature_pps_difference_not_greater_than(condition_value)

    # Act
    result = SingleFeatureContributionTrainTest().run(train_dataset=Dataset(df, label='label'),
                                                      test_dataset=Dataset(df2, label='label'))
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train-Test features\' Predictive Power Score difference is not greater than {condition_value}',
        details='Features with PPS difference above threshold: {\'x2\': \'0.31\'}'
    ))


def test_train_test_condition_pps_train_pass():
    # Arrange
    df, df2, expected = util_generate_second_similar_dataframe_and_expected()
    condition_value = 0.9
    check = SingleFeatureContributionTrainTest().add_condition_feature_pps_in_train_not_greater_than(condition_value)

    # Act
    result = SingleFeatureContributionTrainTest().run(train_dataset=Dataset(df, label='label'),
                                                      test_dataset=Dataset(df2, label='label'))
    condition_result, *_ = check.conditions_decision(result)
    print(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        name=f'Train features\' Predictive Power Score is not greater than {condition_value}'
    ))


def test_train_test_condition_pps_train_fail():
    # Arrange
    df, df2, expected = util_generate_second_similar_dataframe_and_expected()
    condition_value = 0.6
    check = SingleFeatureContributionTrainTest().add_condition_feature_pps_in_train_not_greater_than(condition_value)

    # Act
    result = SingleFeatureContributionTrainTest().run(train_dataset=Dataset(df, label='label'),
                                                      test_dataset=Dataset(df2, label='label'))
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name=f'Train features\' Predictive Power Score is not greater than {condition_value}',
        details='Features in train dataset with PPS above threshold: {\'x2\': \'0.84\'}'
    ))

