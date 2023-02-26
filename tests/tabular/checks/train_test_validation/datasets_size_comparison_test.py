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
from hamcrest import assert_that, has_entries, instance_of

from deepchecks.core import CheckResult, ConditionCategory
from deepchecks.tabular.checks import DatasetsSizeComparison
from tests.base.utils import equal_condition_result


def test_test_dataset_size_check(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model

    check_result = DatasetsSizeComparison().run(train, test, model)

    assert_that(check_result, instance_of(CheckResult))
    assert_that(check_result.value, instance_of(dict))
    assert_that(check_result.value, has_entries({
        "Train": instance_of(int),
        "Test": instance_of(int)
    }))


def test_test_dataset_size_check_with_condition_that_should_pass(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    check = DatasetsSizeComparison().add_condition_test_size_greater_or_equal(10)

    check_result = check.run(train, test, model)
    condition_result, *_ = check_result.conditions_results

    assert_that(condition_result, matcher=equal_condition_result( # type: ignore
        is_pass=True,
        name='Test dataset size is greater or equal to 10',
        details='Test dataset contains 50 samples',
        category=ConditionCategory.PASS
    ))


def test_test_dataset_size_check_with_condition_that_should_not_pass(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    check = DatasetsSizeComparison().add_condition_test_size_greater_or_equal(10_000)

    check_result = check.run(train, test, model)
    condition_result, *_ = check_result.conditions_results

    assert_that(condition_result, matcher=equal_condition_result( # type: ignore
        is_pass=False,
        name='Test dataset size is greater or equal to 10000',
        details='Test dataset contains 50 samples',
        category=ConditionCategory.FAIL
    ))


def test_test_dataset_size_check_with_size_ratio_condition_that_should_pass(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    check = DatasetsSizeComparison().add_condition_test_train_size_ratio_greater_than(0.2)

    check_result = check.run(train, test, model)
    condition_result, *_ = check_result.conditions_results

    assert_that(condition_result, matcher=equal_condition_result( # type: ignore
        is_pass=True,
        name='Test-Train size ratio is greater than 0.2',
        details='Test-Train size ratio is 0.5',
        category=ConditionCategory.PASS
    ))


def test_test_dataset_size_check_with_size_ratio_condition_that_should_not_pass(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    check = DatasetsSizeComparison().add_condition_test_train_size_ratio_greater_than(0.8)

    check_result = check.run(train, test, model)
    condition_result, *_ = check_result.conditions_results

    assert_that(condition_result, matcher=equal_condition_result( # type: ignore
        is_pass=False,
        name='Test-Train size ratio is greater than 0.8',
        details=r'Test-Train size ratio is 0.5',
        category=ConditionCategory.FAIL
    ))


def test_condition_train_not_smaller_than_test_pass(iris):
    # Arrange
    train = iris[:100]
    test = iris[100:]
    check = DatasetsSizeComparison().add_condition_train_dataset_greater_or_equal_test()

    # Act
    check_result = check.run(train, test)
    condition_result, *_ = check_result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Train dataset is larger than test dataset by +50 samples',
        name='Train dataset is greater or equal to test dataset'
    ))


def test_condition_train_not_smaller_than_test_fail(iris):
    # Arrange
    train = iris[100:]
    test = iris[:100]
    check = DatasetsSizeComparison().add_condition_train_dataset_greater_or_equal_test()

    # Act
    check_result = check.run(train, test)
    condition_result, *_ = check_result.conditions_results

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='Train dataset is greater or equal to test dataset',
        details='Train dataset is smaller than test dataset by -50 samples'
    ))
