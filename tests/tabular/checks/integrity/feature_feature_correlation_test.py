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
"""Tests for Feature Feature Correlation check"""
from hamcrest import (assert_that, calling, contains_exactly, contains_inanyorder, equal_to, has_items, has_length,
                      raises)

from deepchecks.tabular.checks.data_integrity.feature_feature_correlation import FeatureFeatureCorrelation
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_feature_feature_correlation(adult_no_split):
    expected_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
       'hours-per-week', 'workclass', 'education', 'marital-status',
       'occupation', 'relationship', 'race', 'sex', 'native-country']
    result = FeatureFeatureCorrelation().run(adult_no_split)
    assert_that(result.value.index, contains_inanyorder(*expected_features))
    assert_that(result.value.columns, contains_inanyorder(*expected_features))
    assert_that(result.display, has_length(3))


def test_feature_feature_correlation_without_display(adult_no_split):
    expected_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
       'hours-per-week', 'workclass', 'education', 'marital-status',
       'occupation', 'relationship', 'race', 'sex', 'native-country']
    result = FeatureFeatureCorrelation(n_samples=9999999).run(adult_no_split, with_display=False)
    assert_that(result.value.index, contains_inanyorder(*expected_features))
    assert_that(result.value.columns, contains_inanyorder(*expected_features))
    assert_that(result.display, has_length(0))


def test_feature_feature_correlation_with_mixed_data(df_with_mixed_datatypes_and_missing_values):
    # This dataset has mixed data types, missing values and a datetime column.
    ds = Dataset(df_with_mixed_datatypes_and_missing_values, cat_features=['cat', 'dog', 'owl'], label='target')
    check = FeatureFeatureCorrelation()
    result = check.run(ds)
    # assert that the datetime column is ignored
    expected_features = ['blue', 'red', 'green', 'black', 'white', 'owl', 'cat', 'dog']
    assert_that(result.value.index, contains_inanyorder(*expected_features))

    # assert that ignored columns are not in the correlation matrix
    check.ignore_columns = ['green', 'dog', 'owl']
    expected_features = ['cat', 'blue', 'red', 'white', 'black']
    result = check.run(ds)
    assert_that(result.value.index, contains_inanyorder(*expected_features))
    assert_that(result.value.columns, contains_inanyorder(*expected_features))

    # assert that has display for top 5 (columns without NaNs), but all the features are in the value
    check = FeatureFeatureCorrelation(show_n_top_columns=5)
    result = check.run(ds)
    expected_features = ['red', 'blue', 'green', 'white', 'black', 'cat', 'dog', 'owl']
    assert_that(result.value.index, contains_exactly(*expected_features))
    assert_that(result.value.columns, contains_exactly(*expected_features))
    assert_that(result.have_display(), equal_to(True))


def test_feature_feature_correlation_pass_condition(adult_no_split):
    high_pairs = [('education', 'education-num')]
    threshold = 0.9
    num_pairs = 1
    check = FeatureFeatureCorrelation()
    result = check.add_condition_max_number_of_pairs_above_threshold(threshold, num_pairs).run(adult_no_split)
    assert_that(result.conditions_results, has_items(
                equal_condition_result(is_pass=True,
                                       details=f'All correlations are less than {threshold} except pairs {high_pairs}',
                                       name=f'Not more than {num_pairs} pairs are correlated above {threshold}')
            ))


def test_feature_feature_correlation_fail_condition(adult_no_split):
    threshold = 0.5
    num_pairs = 3
    high_pairs = [('age', 'marital-status'), ('education-num', 'occupation'), ('education', 'education-num'),
                  ('marital-status', 'relationship')]
    check = FeatureFeatureCorrelation()
    result = check.add_condition_max_number_of_pairs_above_threshold(threshold, num_pairs).run(adult_no_split)

    assert_that(result.conditions_results, has_items(
                equal_condition_result(is_pass=False,
                                       details=f'Correlation is greater than {threshold} for pairs {high_pairs}',
                                       name=f'Not more than {num_pairs} pairs are correlated above {threshold}')
            ))
