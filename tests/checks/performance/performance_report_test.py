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
    assert_that(calling(PerformanceReport().run).with_args(bad_dataset, None),
                raises(DeepchecksValueError,
                       'Check requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(calling(PerformanceReport().run).with_args(iris_dataset, iris_adaboost),
                raises(DeepchecksValueError, 'Check requires dataset to have a label column'))


def test_classification(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = PerformanceReport()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert_that(result, has_entries({
        'Accuracy (Default)': close_to(0.96, 0.01),
        'Precision - Macro Average (Default)': close_to(0.96, 0.01),
        'Recall - Macro Average (Default)': close_to(0.96, 0.01)
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
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert_that(result, has_entries({
        'Accuracy (Default)': close_to(0.96, 0.01),
        'Precision - Macro Average (Default)': close_to(0.96, 0.01),
        'Recall - Macro Average (Default)': close_to(0.96, 0.01)
    }))


def test_classification_nan_labels(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = PerformanceReport()
    data_with_nan = iris_labeled_dataset.data.copy()
    data_with_nan[iris_labeled_dataset.label_name].iloc[0] = float('nan')
    iris_labeled_dataset = Dataset(data_with_nan,
                                   label_name=iris_labeled_dataset.label_name)
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert_that(result, has_entries({
        'Accuracy (Default)': close_to(0.96, 0.01),
        'Precision - Macro Average (Default)': close_to(0.96, 0.01),
        'Recall - Macro Average (Default)': close_to(0.96, 0.01)
    }))


def test_regression(diabetes, diabetes_model):
    # Arrange
    _, validation = diabetes
    check = PerformanceReport()
    # Act X
    result = check.run(validation, diabetes_model).value
    # Assert
    assert_that(result, has_entries({
        'Neg RMSE (Default)': close_to(-50, 20),
        'Neg MAE (Default)': close_to(-45, 20),
    }))


def test_condition_min_score_not_passed(diabetes, diabetes_model):
    # Arrange
    _, validation = diabetes
    check = PerformanceReport().add_condition_score_not_less_than(-50)
    # Act X
    result: List[ConditionResult] = check.conditions_decision(check.run(validation, diabetes_model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details=re.compile('Scores that did not pass threshold: \\{\'Neg RMSE \(Default\)\':'),
                               name='Score is not less than -50')
    ))


def test_condition_min_score_passed(diabetes, diabetes_model):
    # Arrange
    _, validation = diabetes
    check = PerformanceReport().add_condition_score_not_less_than(-5_000)
    # Act X
    result: List[ConditionResult] = check.conditions_decision(check.run(validation, diabetes_model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Score is not less than -5000')
    ))
