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
"""Contains unit tests for the RegressionErrorDistribution check."""
from hamcrest import assert_that, calling, close_to, greater_than, has_items, has_length, raises

from deepchecks.core import ConditionCategory
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError, ModelValidationError
from deepchecks.tabular.checks.model_evaluation import RegressionErrorDistribution
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(RegressionErrorDistribution().run).with_args(bad_dataset, None),
        raises(DeepchecksValueError, 'non-empty instance of Dataset or DataFrame was expected, instead got str')
    )


def test_dataset_no_label(diabetes_dataset_no_label, diabetes_model):
    # Assert
    assert_that(
        calling(RegressionErrorDistribution().run).with_args(diabetes_dataset_no_label, diabetes_model),
        raises(DeepchecksNotSupportedError,
               'Dataset does not contain a label column')
    )


def test_multiclass_model(iris_split_dataset_and_model):
    # Assert
    _, test, clf = iris_split_dataset_and_model
    assert_that(
        calling(RegressionErrorDistribution().run).with_args(test, clf),
        raises(ModelValidationError, 'Check is irrelevant for classification tasks'))


def test_regression_error_distribution(diabetes_split_dataset_and_model):
    # Arrange
    _, test, clf = diabetes_split_dataset_and_model
    check = RegressionErrorDistribution()
    # Act X
    result = check.run(test, clf)
    # Assert
    assert_that(result.value['Kurtosis Value'], close_to(0.028, 0.001))
    assert_that(result.display, has_length(greater_than(0)))


def test_regression_error_distribution_without_display(diabetes_split_dataset_and_model):
    # Arrange
    _, test, clf = diabetes_split_dataset_and_model
    check = RegressionErrorDistribution()
    # Act X
    result = check.run(test, clf, with_display=False)
    # Assert
    assert_that(result.value['Kurtosis Value'], close_to(0.028, 0.001))
    assert_that(result.display, has_length(0))


def test_condition_absolute_kurtosis_not_greater_than_not_passed(diabetes_split_dataset_and_model):
    # Arrange
    _, test, clf = diabetes_split_dataset_and_model
    test = Dataset(test.data.copy(), label='target')
    test._data[test.label_name] =300

    check = RegressionErrorDistribution().add_condition_kurtosis_greater_than()

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Kurtosis value higher than -0.1',
                               details='Found kurtosis value of -0.92572',
                               category=ConditionCategory.WARN)
    ))


def test_condition_absolute_kurtosis_not_greater_than_passed(diabetes_split_dataset_and_model):
    _, test, clf = diabetes_split_dataset_and_model

    check = RegressionErrorDistribution().add_condition_kurtosis_greater_than()

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Found kurtosis value of 0.02867',
                               name='Kurtosis value higher than -0.1')
    ))


def test_condition_absolute_kurtosis_not_greater_than_not_passed_0_max(diabetes_split_dataset_and_model):
    _, test, clf = diabetes_split_dataset_and_model

    check = RegressionErrorDistribution().add_condition_kurtosis_greater_than(threshold=1)

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Kurtosis value higher than 1',
                               details='Found kurtosis value of 0.02867',
                               category=ConditionCategory.WARN)
    ))


def test_regression_error(diabetes_split_dataset_and_model):
    # Arrange
    _, test, clf = diabetes_split_dataset_and_model
    check = RegressionErrorDistribution()
    # Act X
    result = check.run(test, clf)
    # Assert
    assert_that(result.value['RMSE'], close_to(57.5, 0.1))
    assert_that(result.value['Mean Prediction Error'], close_to(-0.008, 0.001))
    assert_that(result.display, has_length(greater_than(0)))


def test_regression_error_without_display(diabetes_split_dataset_and_model):
    # Arrange
    _, test, clf = diabetes_split_dataset_and_model
    check = RegressionErrorDistribution()
    # Act X
    result = check.run(test, clf, with_display=False)
    # Assert
    assert_that(result.value['RMSE'], close_to(57.5, 0.1))
    assert_that(result.value['Mean Prediction Error'], close_to(-0.008, 0.001))
    assert_that(result.display, has_length(0))


def test_condition_error_ratio_not_greater_than_not_passed(diabetes_split_dataset_and_model):
    # Arrange
    _, test, clf = diabetes_split_dataset_and_model
    test = Dataset(test.data.copy(), label='target')
    test._data[test.label_name] = 300

    check = RegressionErrorDistribution().add_condition_systematic_error_ratio_to_rmse_less_than()

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Systematic error ratio lower than 0.01',
                               details='Found systematic error to rmse ratio of 0.93')
    ))


def test_condition_error_ratio_not_greater_than_passed(diabetes_split_dataset_and_model):
    _, test, clf = diabetes_split_dataset_and_model

    check = RegressionErrorDistribution().add_condition_systematic_error_ratio_to_rmse_less_than()

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Found systematic error to rmse ratio of 1.40E-4',
                               name='Systematic error ratio lower than 0.01')
    ))


def test_condition_error_ratio_not_greater_than_not_passed_0_max(diabetes_split_dataset_and_model):
    _, test, clf = diabetes_split_dataset_and_model

    check = RegressionErrorDistribution().add_condition_systematic_error_ratio_to_rmse_less_than(max_ratio=0)

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Systematic error ratio lower than 0',
                               details='Found systematic error to rmse ratio of 1.40E-4')
    ))
