# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""Test functions of trust score comparison."""
from hamcrest import assert_that, has_entries, close_to, calling, raises

from deepchecks import CheckResult
from deepchecks.checks import TrustScoreComparison
from deepchecks.errors import DeepchecksValueError
from tests.checks.utils import equal_condition_result


def test_trust_score_comparison(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = TrustScoreComparison(min_test_samples=50)

    # Act
    result = check.run(train, test, model)

    # Assert
    assert_that(result.value, has_entries({
        'train': close_to(5.78, 0.01),
        'test': close_to(4.49, 0.01)
    }))


def test_condition_mean_score_percent_decline_fail():
    # Arrange
    check = TrustScoreComparison().add_condition_mean_score_percent_decline_not_greater_than(0.1)
    result = CheckResult({'train': 1.8, 'test': 1.2})

    # Act
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='Mean trust score decline is not greater than 10.00%',
        details='Found decline of: -33.33%'
    ))


def test_condition_mean_score_percent_decline_pass():
    # Arrange
    check = TrustScoreComparison().add_condition_mean_score_percent_decline_not_greater_than(0.4)
    result = CheckResult({'train': 1.8, 'test': 1.2})

    # Act
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        name='Mean trust score decline is not greater than 40.00%'
    ))


def test_sample_size_too_small(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = TrustScoreComparison(min_test_samples=500)

    assert_that(calling(check.run).with_args(train, test, model),
                raises(DeepchecksValueError, 'Number of samples in test dataset have not passed the minimum'))


def test_regression_model_fail(diabetes_split_dataset_and_model):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model
    check = TrustScoreComparison(min_test_samples=50)

    assert_that(calling(check.run).with_args(train, test, model),
                raises(DeepchecksValueError, 'Check supports only classification'))
