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
"""Tests for Feature Feature Correlation check"""
from hamcrest import assert_that, contains_exactly, has_items, equal_to

from deepchecks.tabular.checks.data_integrity.feature_feature_correlation import FeatureFeatureCorrelation
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_feature_feature_correlation(adult_no_split):
    expected_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
       'hours-per-week', 'workclass', 'education', 'marital-status',
       'occupation', 'relationship', 'race', 'sex', 'native-country']
    result = FeatureFeatureCorrelation().run(adult_no_split)
    assert_that(result.value.index, contains_exactly(*expected_features))
    assert_that(result.value.columns, contains_exactly(*expected_features))


def test_feature_feature_correlation_corrupted_data(df_with_single_nans_in_different_rows):
    ds = Dataset(df_with_single_nans_in_different_rows)
    check = FeatureFeatureCorrelation()
    result = check.run(ds)
    assert_that(len(result.value.index), equal_to(2))


def test_feature_feature_correlation_pass_condition(adult_no_split):
    high_pairs = [('education-num', 'education')]
    threshold = 0.9
    num_pairs = 1
    check = FeatureFeatureCorrelation()
    result = check.add_condition_max_number_of_pairs_above(threshold, num_pairs).run(adult_no_split)
    assert_that(result.conditions_results, has_items(
                equal_condition_result(is_pass=True,
                                       details=f'All correlations are less than {threshold} except pairs {high_pairs}',
                                       name=f'Not more than {num_pairs} pairs are correlated above {threshold}')
            ))


def test_feature_feature_correlation_fail_condition(adult_no_split):
    threshold = 0.5
    num_pairs = 3
    high_pairs = [('age', 'marital-status'), ('education-num', 'education'), ('education-num', 'occupation'),
                  ('marital-status', 'relationship')]
    check = FeatureFeatureCorrelation()
    result = check.add_condition_max_number_of_pairs_above(threshold, num_pairs).run(adult_no_split)

    assert_that(result.conditions_results, has_items(
                equal_condition_result(is_pass=False,
                                       details=f'Correlation is greater than {threshold} for pairs {high_pairs}',
                                       name=f'Not more than {num_pairs} pairs are correlated above {threshold}')
            ))




