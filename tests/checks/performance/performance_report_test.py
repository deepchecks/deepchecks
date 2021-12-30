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
"""Contains unit tests for the performance report check."""
import re
from typing import List
from hamcrest import assert_that, calling, raises, close_to, has_entries, has_items
from sklearn.ensemble import AdaBoostClassifier

from deepchecks import ConditionResult, Dataset
from deepchecks.checks.performance import PerformanceReport
from deepchecks.errors import DeepchecksValueError

from tests.checks.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(PerformanceReport().run).with_args(bad_dataset, None, None),
                raises(DeepchecksValueError,
                       'Check requires dataset to be of type Dataset. instead got: str'))


def test_model_wrong_input(iris_labeled_dataset):
    bad_model = 'wrong_input'
    # Act & Assert
    assert_that(calling(PerformanceReport().run).with_args(iris_labeled_dataset, iris_labeled_dataset,
                                                                    bad_model),
                raises(DeepchecksValueError,
                       'Model must inherit from one of supported models: .*'))


def test_dataset_no_label(iris_dataset):
    # Assert
    assert_that(calling(PerformanceReport().run).with_args(iris_dataset, iris_dataset, None),
                raises(DeepchecksValueError, 'Check requires dataset to have a label column'))


def test_dataset_no_shared_label(iris_labeled_dataset):
    # Assert
    iris_dataset_2 = Dataset(iris_labeled_dataset.data, label_name='sepal length (cm)')
    assert_that(calling(PerformanceReport().run).with_args(iris_labeled_dataset, iris_dataset_2, None),
                raises(DeepchecksValueError,
                       'Check requires datasets to share the same label'))

def test_classification(iris_split_dataset_and_model):
    # Arrange
    train, test, model = iris_split_dataset_and_model
    check = PerformanceReport()
    # Act X
    result = check.run(train, test, model).value
    # Assert
    assert_that(result, has_entries({
        'Accuracy': close_to(0.96, 0.01),
        'Precision - Macro Average': close_to(0.96, 0.01),
        'Recall - Macro Average': close_to(0.96, 0.01)
    }))


def test_classification_string_labels(iris_labeled_dataset):
    # Arrange
    check = PerformanceReport()
    replace_dict = {iris_labeled_dataset.label_name: {0: 'b', 1: 'e', 2: 'a'}}
    iris_labeled_dataset = Dataset(iris_labeled_dataset.data.replace(replace_dict),
                                   label_name=iris_labeled_dataset.label_name)

    iris_adaboost = AdaBoostClassifier(random_state=0)
    iris_adaboost.fit(iris_labeled_dataset.features_columns, iris_labeled_dataset.label_col)
    # Act X
    result = check.run(iris_labeled_dataset, iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert_that(result, has_entries({
        'Accuracy': close_to(0.96, 0.01),
        'Precision - Macro Average': close_to(0.96, 0.01),
        'Recall - Macro Average': close_to(0.96, 0.01)
    }))


def test_classification_nan_labels(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = PerformanceReport()
    data_with_nan = iris_labeled_dataset.data.copy()
    data_with_nan[iris_labeled_dataset.label_name].iloc[0] = float('nan')
    iris_labeled_dataset = Dataset(data_with_nan,
                                   label_name=iris_labeled_dataset.label_name)
    # Act X
    result = check.run(iris_labeled_dataset, iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert_that(result, has_entries({
        'Accuracy': close_to(0.96, 0.01),
        'Precision - Macro Average': close_to(0.96, 0.01),
        'Recall - Macro Average': close_to(0.96, 0.01)
    }))


def test_regression(diabetes_split_dataset_and_model):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model
    check = PerformanceReport()
    # Act X
    result = check.run(train, test, model).value
    # Assert
    assert_that(result, has_entries({
        'RMSE': close_to(-50, 20),
        'MAE': close_to(-45, 20),
    }))


def test_condition_max_score_not_passed(diabetes_split_dataset_and_model):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model
    check = PerformanceReport().add_condition_score_not_greater_than(50)
    # Act X
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details=re.compile('Scores that passed threshold:<br>\\[\\{\'Dataset\': '
                                                  '\'Test\', \'Metric\': \'RMSE\', \'Value\': '),
                               name='Scores are not greater than 50')
    ))


def test_condition_max_score_passed(diabetes_split_dataset_and_model):
    # Arrange
    train, test, model = diabetes_split_dataset_and_model
    check = PerformanceReport().add_condition_score_not_greater_than(5_000)
    # Act X
    result: List[ConditionResult] = check.conditions_decision(check.run(train, test, model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Scores are not greater than 5000')
    ))
