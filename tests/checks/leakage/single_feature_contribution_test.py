"""Contains unit tests for the single_feature_contribution check."""
import numpy as np
import pandas as pd

from deepchecks import Dataset
from deepchecks.checks.leakage import SingleFeatureContribution, \
                                    SingleFeatureContributionTrainTest
from deepchecks.utils import DeepchecksValueError

from hamcrest import assert_that, is_in, close_to, calling, raises, equal_to


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

    return df, df2, {'x1': 0.0, 'x2': 0.3, 'x3': 0.0, 'x4': 0.0, 'x5': 0.0}


def test_assert_single_feature_contribution():
    df, expected = util_generate_dataframe_and_expected()
    result = SingleFeatureContribution().run(dataset=Dataset(df, label='label'))
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_show_top_single_feature_contribution():
    df, expected = util_generate_dataframe_and_expected()
    result = SingleFeatureContribution(n_show_top=3).run(dataset=Dataset(df, label='label'))
    assert_that(len(result.value), equal_to(3))
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))

def test_dataset_wrong_input():
    wrong = 'wrong_input'
    assert_that(
        calling(SingleFeatureContribution().run).with_args(wrong),
        raises(DeepchecksValueError, 'Check SingleFeatureContribution requires dataset to be of type Dataset. '
                                   'instead got: str'))


def test_dataset_no_label():
    df, _ = util_generate_dataframe_and_expected()
    df = Dataset(df)
    assert_that(
        calling(SingleFeatureContribution().run).with_args(dataset=df),
        raises(DeepchecksValueError, 'Check SingleFeatureContribution requires dataset to have a label column'))


def test_trainval_assert_single_feature_contribution():
    df, df2, expected = util_generate_second_similar_dataframe_and_expected()
    result = SingleFeatureContributionTrainTest().run(train_dataset=Dataset(df, label='label'),
                                                      test_dataset=Dataset(df2, label='label'))
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_trainval_show_top_single_feature_contribution():
    df, df2, expected = util_generate_second_similar_dataframe_and_expected()
    result = SingleFeatureContributionTrainTest(n_show_top=3).run(train_dataset=Dataset(df, label='label'),
                                                                  test_dataset=Dataset(df2, label='label'))
    assert_that(len(result.value), equal_to(3))
    for key, value in result.value.items():
        assert_that(key, is_in(expected.keys()))
        assert_that(value, close_to(expected[key], 0.1))


def test_trainval_dataset_wrong_input():
    wrong = 'wrong_input'
    assert_that(
        calling(SingleFeatureContributionTrainTest().run).with_args(wrong, wrong),
        raises(DeepchecksValueError,
               'Check SingleFeatureContributionTrainTest requires dataset to be of type Dataset. '
               'instead got: str'))


def test_trainval_dataset_no_label():
    df, df2, _ = util_generate_second_similar_dataframe_and_expected()
    assert_that(
        calling(SingleFeatureContributionTrainTest().run).with_args(train_dataset=Dataset(df),
                                                                    test_dataset=Dataset(df2)),
        raises(DeepchecksValueError,
               'Check SingleFeatureContributionTrainTest requires dataset to have a label column'))


def test_trainval_dataset_diff_columns():
    df, df2, _ = util_generate_second_similar_dataframe_and_expected()
    df = df.rename({'x2': 'x6'}, axis=1)
    assert_that(
        calling(SingleFeatureContributionTrainTest().run)
            .with_args(train_dataset=Dataset(df, label='label'),
                       test_dataset=Dataset(df2, label='label')),
        raises(DeepchecksValueError,
               'Check SingleFeatureContributionTrainTest requires datasets to share the same features'))
