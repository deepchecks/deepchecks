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
"""Contains unit tests for the RegressionErrorDistribution check."""
from hamcrest import assert_that, calling, raises, has_items, close_to

from deepchecks.core import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError, ModelValidationError, DeepchecksNotSupportedError
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks.performance import RegressionErrorDistribution
from tests.checks.utils import equal_condition_result


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
               'There is no label defined to use. Did you pass a DataFrame instead of a Dataset?')
    )


def test_multiclass_model(iris_split_dataset_and_model):
    # Assert
    _, test, clf = iris_split_dataset_and_model
    assert_that(
        calling(RegressionErrorDistribution().run).with_args(test, clf),
        raises(
            ModelValidationError,
            r'Check is relevant for models of type \[\'regression\'\], '
            r'but received model of type \'multiclass\'')
    )


def test_regression_error_distribution(diabetes_split_dataset_and_model):
    # Arrange
    _, test, clf = diabetes_split_dataset_and_model
    check = RegressionErrorDistribution()
    # Act X
    result = check.run(test, clf).value
    # Assert
    assert_that(result, close_to(0.028, 0.001))


def test_condition_absolute_kurtosis_not_greater_than_not_passed(diabetes_split_dataset_and_model):
    # Arrange
    _, test, clf = diabetes_split_dataset_and_model
    test = Dataset(test.data.copy(), label='target')
    test._data[test.label_name] =300

    check = RegressionErrorDistribution().add_condition_kurtosis_not_less_than()

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Kurtosis value is not less than -0.1',
                               details='Found kurtosis below threshold: -0.92572',
                               category=ConditionCategory.WARN)
    ))


def test_condition_absolute_kurtosis_not_greater_than_passed(diabetes_split_dataset_and_model):
    _, test, clf = diabetes_split_dataset_and_model

    check = RegressionErrorDistribution().add_condition_kurtosis_not_less_than()

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Kurtosis value is not less than -0.1')
    ))


def test_condition_absolute_kurtosis_not_greater_than_not_passed_0_max(diabetes_split_dataset_and_model):
    _, test, clf = diabetes_split_dataset_and_model

    check = RegressionErrorDistribution().add_condition_kurtosis_not_less_than(min_kurtosis=1)

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Kurtosis value is not less than 1',
                               details='Found kurtosis below threshold: 0.02867',
                               category=ConditionCategory.WARN)
    ))
