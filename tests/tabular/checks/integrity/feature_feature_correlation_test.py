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
import pandas as pd
from hamcrest import assert_that, has_items, close_to

from deepchecks.tabular.checks.data_integrity.feature_feature_correlation import FeatureFeatureCorrelation
from deepchecks.tabular.datasets.classification import adult
from tests.base.utils import equal_condition_result


ds = adult.load_data(as_train_test=False)


def test_feature_feature_correlation():
    result = FeatureFeatureCorrelation().run(ds)


def test_feature_feature_correlation_pass_condition():
    high_pairs = [('education-num', 'education')]
    threshold = 0.9
    num_pairs = 1
    check = FeatureFeatureCorrelation()
    result = check.add_condition_all_correlations_less_than(threshold, num_pairs).run(ds)
    assert_that(result.conditions_results, has_items(
                equal_condition_result(is_pass=True,
                                       details=f'All correlations are less than {threshold} except pairs {high_pairs}',
                                       name=f'Not more than {num_pairs} pairs are correlated above {threshold}')
            ))


def test_feature_feature_correlation_fail_condition():
    threshold = 0.5
    num_pairs = 3
    high_pairs = [('age', 'marital-status'), ('education-num', 'education'), ('education-num', 'occupation'),
                  ('marital-status', 'relationship')]
    check = FeatureFeatureCorrelation()
    result = check.add_condition_all_correlations_less_than(threshold, num_pairs).run(ds)

    assert_that(result.conditions_results, has_items(
                equal_condition_result(is_pass=False,
                                       details=f'Correlation is greater than {threshold} for pairs {high_pairs}',
                                       name=f'Not more than {num_pairs} pairs are correlated above {threshold}')
            ))
