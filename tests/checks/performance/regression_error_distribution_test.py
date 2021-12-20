# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
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

from deepchecks.base import Dataset
from deepchecks.checks.performance import RegressionErrorDistribution
from deepchecks.errors import DeepchecksValueError
from tests.checks.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(RegressionErrorDistribution().run).with_args(bad_dataset, None),
                raises(DeepchecksValueError,
                       'Check requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(diabetes_df, diabetes_model):
    # Assert
    assert_that(calling(RegressionErrorDistribution().run).with_args(Dataset(diabetes_df), diabetes_model),
                raises(DeepchecksValueError, 'Check requires dataset to have a label column'))


def test_multiclass_model(iris_split_dataset_and_model):
    # Assert
    _, test, clf = iris_split_dataset_and_model
    assert_that(calling(RegressionErrorDistribution().run).with_args(test, clf),
                raises(DeepchecksValueError, r'Expected model to be a type from'
                                           r' \[\'regression\'\], but received model of type: multiclass'))


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
    test = Dataset(test.data.copy(), label_name='target')
    test._data[test.label_name] =300

    check = RegressionErrorDistribution().add_condition_kurtosis_not_less_than()

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Kurtosis value not less than -0.1',
                               details='kurtosis: -0.92572')
    ))


def test_condition_absolute_kurtosis_not_greater_than_passed(diabetes_split_dataset_and_model):
    _, test, clf = diabetes_split_dataset_and_model

    check = RegressionErrorDistribution().add_condition_kurtosis_not_less_than()

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Kurtosis value not less than -0.1')
    )) 

def test_condition_absolute_kurtosis_not_greater_than_not_passed_0_max(diabetes_split_dataset_and_model):
    _, test, clf = diabetes_split_dataset_and_model

    check = RegressionErrorDistribution().add_condition_kurtosis_not_less_than(min_kurtosis=1)

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Kurtosis value not less than 1',
                               details='kurtosis: 0.02867')
    )) 
