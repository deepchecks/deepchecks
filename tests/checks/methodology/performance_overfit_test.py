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
"""Contains unit tests for the performance overfit check."""
import pandas as pd

from deepchecks import Dataset
from deepchecks.base.check import CheckResult
from deepchecks.checks import TrainTestDifferenceOverfit
from deepchecks.errors import DeepchecksValueError
from hamcrest import assert_that, calling, raises, close_to, has_entries

from tests.checks.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(TrainTestDifferenceOverfit().run).with_args(bad_dataset, None, None),
                raises(DeepchecksValueError,
                       'Check requires dataset to be of type Dataset. instead got: str'))


def test_model_wrong_input(iris_labeled_dataset):
    bad_model = 'wrong_input'
    # Act & Assert
    assert_that(calling(TrainTestDifferenceOverfit().run).with_args(iris_labeled_dataset, iris_labeled_dataset,
                                                                    bad_model),
                raises(DeepchecksValueError,
                       'Model must inherit from one of supported models: .*'))


def test_dataset_no_label(iris_dataset):
    # Assert
    assert_that(calling(TrainTestDifferenceOverfit().run).with_args(iris_dataset, iris_dataset, None),
                raises(DeepchecksValueError, 'Check requires dataset to have a label column'))


def test_dataset_no_shared_label(iris_labeled_dataset):
    # Assert
    iris_dataset_2 = Dataset(iris_labeled_dataset.data, label_name='sepal length (cm)')
    assert_that(calling(TrainTestDifferenceOverfit().run).with_args(iris_labeled_dataset, iris_dataset_2, None),
                raises(DeepchecksValueError,
                       'Check requires datasets to share the same label'))


def test_dataset_no_shared_features(iris_labeled_dataset):
    # Arrange
    iris_dataset_2 = Dataset(pd.concat(
        [iris_labeled_dataset.data,
         iris_labeled_dataset.data[['sepal length (cm)']].rename(columns={'sepal length (cm)': '1'})],
        axis=1),
        label_name=iris_labeled_dataset.label_name)
    # Assert
    assert_that(calling(TrainTestDifferenceOverfit().run).with_args(iris_labeled_dataset, iris_dataset_2, None),
                raises(DeepchecksValueError,
                       'Check requires datasets to share the same features'))


def test_no_diff(iris_split_dataset_and_model):
    # Arrange
    train, _, model = iris_split_dataset_and_model
    check_obj = TrainTestDifferenceOverfit()
    # Act
    result = check_obj.run(train, train, model)

    # Assert
    train = result.value['train']
    test = result.value['test']
    expected = {'Accuracy - Default': close_to(0.96, 0.01), 'Precision - Macro Average - Default': close_to(0.96, 0.01),
                'Recall - Macro Average - Default': close_to(0.96, 0.01)}
    assert_that(train, has_entries(expected))
    assert_that(test, has_entries(expected))


def test_with_diff(iris_split_dataset_and_model):
    # Arrange
    train, val, model = iris_split_dataset_and_model
    check_obj = TrainTestDifferenceOverfit()

    # Act
    result = check_obj.run(train, val, model)

    # Assert
    train = result.value['train']
    test = result.value['test']
    expected_train = {'Accuracy - Default': close_to(0.96, 0.01),
                      'Precision - Macro Average - Default': close_to(0.96, 0.01),
                      'Recall - Macro Average - Default': close_to(0.96, 0.01)}
    expected_test = {'Accuracy - Default': close_to(0.92, 0.01),
                     'Precision - Macro Average - Default': close_to(0.92, 0.01),
                     'Recall - Macro Average - Default': close_to(0.92, 0.01)}
    assert_that(train, has_entries(expected_train))
    assert_that(test, has_entries(expected_test))


def test_custom_metrics(iris_split_dataset_and_model):
    # Arrange
    train, val, model = iris_split_dataset_and_model
    check_obj = TrainTestDifferenceOverfit(
        alternative_scorers={'Accuracy': 'accuracy', 'Always 0.5': lambda x, y, z: 0.5}
    )

    # Act
    result = check_obj.run(train, val, model)

    # Assert
    train = result.value['train']
    test = result.value['test']
    expected_train = {'Accuracy': close_to(0.96, 0.01), 'Always 0.5': 0.5}
    expected_test = {'Accuracy': close_to(0.92, 0.01), 'Always 0.5': 0.5}
    assert_that(train, has_entries(expected_train))
    assert_that(test, has_entries(expected_test))


def test_train_test_difference_condition_that_should_pass():
    """
    Testing condition generated by `TrainTestDifferenceOverfit.add_condition_train_test_difference_not_greater_than` method
    """
    # Arrange
    check = TrainTestDifferenceOverfit().add_condition_difference_not_greater_than(0.1)
    condition_satisfying = {
        'train': {'x1': 0.88, 'x2': 0.64, 'x3': 0.71},
        'test': {'x1': 0.88, 'x2': 0.64, 'x3': 0.71},
    }

    # Act
    condition_result, *_ = check.conditions_decision(CheckResult(condition_satisfying))

    # Assert
    assert_that(condition_result, equal_condition_result(
            is_pass=True,
            name='Train-Test scores difference is not greater than 0.1'
    ))


def test_train_test_difference_condition_that_should_not_pass():
    """
    Testing condition generated by `TrainTestDifferenceOverfit.add_condition_train_test_difference_not_greater_than` method
    """
    # Arrange
    check = TrainTestDifferenceOverfit().add_condition_difference_not_greater_than(0.1)
    condition_satisfying = {
        'train': {'x1': 0.88, 'x2': 0.64, 'x3': 0.71},
        'test':  {'x1': 0.5, 'x2': 0.2, 'x3': 0.3},
    }

    # Act
    condition_result, *_ = check.conditions_decision(CheckResult(condition_satisfying))

    # Assert
    assert_that(condition_result, equal_condition_result(
            is_pass=False,
            name='Train-Test scores difference is not greater than 0.1',
            details="Found performance degradation in: x1 (train=88.00% test=50.00%), "
                    "x2 (train=64.00% test=20.00%), x3 (train=71.00% test=30.00%)"
    ))


def test_train_test_percentage_degradation_condition_that_should_pass():
    """
    Testing condition generated by `TrainTestDifferenceOverfit.add_condition_train_test_ratio_not_greater_than` method
    """
    # Arrange
    check = TrainTestDifferenceOverfit().add_condition_degradation_ratio_not_greater_than(0.1)
    condition_satisfying = {
        'train': {'x1': 0.88, 'x2': 0.64, 'x3': 0.71},
        'test':   {'x1': 0.87, 'x2': 0.66, 'x3': 0.71},
    }

    # Act
    condition_result, *_ = check.conditions_decision(CheckResult(condition_satisfying))

    # Assert
    assert_that(condition_result, equal_condition_result(
            is_pass=True,
            name='Train-Test scores degradation ratio is not greater than 0.1'
    ))


def test_train_test_ratio_condition_that_should_not_pass():
    """
    Testing condition generated by `TrainTestDifferenceOverfit.add_condition_train_test_ratio_not_greater_than` method
    """
    # Arrange
    check = TrainTestDifferenceOverfit().add_condition_degradation_ratio_not_greater_than(0.2)
    condition_satisfying = {
        'train': {'x1': 0.88, 'x2': 0.64, 'x3': 0.71},
        'test': {'x1': 0.41, 'x2': 0.5, 'x3': 0.65},
    }

    # Act
    condition_result, *_ = check.conditions_decision(CheckResult(condition_satisfying))

    # Assert
    assert_that(condition_result, equal_condition_result(
            is_pass=False,
            name='Train-Test scores degradation ratio is not greater than 0.2',
            details="Found performance degradation in: x1 (train=88.00% test=41.00%), x2 (train=64.00% test=50.00%)"
    ))
