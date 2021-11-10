"""Contains unit tests for the performance report check."""
import re
from typing import List

from mlchecks import ConditionResult
from mlchecks.checks.performance import PerformanceReport
from mlchecks.utils import MLChecksValueError
from hamcrest import assert_that, calling, raises, close_to, has_entries, has_items

from tests.checks.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(PerformanceReport().run).with_args(bad_dataset, None),
                raises(MLChecksValueError,
                       'Check PerformanceReport requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(calling(PerformanceReport().run).with_args(iris_dataset, iris_adaboost),
                raises(MLChecksValueError, 'Check PerformanceReport requires dataset to have a label column'))


def test_classification(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = PerformanceReport()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert_that(result, has_entries({
        'Accuracy': close_to(0.96, 0.01),
        'Precision - Macro Average': close_to(0.96, 0.01),
        'Recall - Macro Average': close_to(0.96, 0.01)
    }))


def test_regression(diabetes, diabetes_model):
    # Arrange
    _, validation = diabetes
    check = PerformanceReport()
    # Act X
    result = check.run(validation, diabetes_model).value
    # Assert
    assert_that(result, has_entries({
        'RMSE': close_to(-50, 20),
        'MSE': close_to(-3200, 1000),
    }))


def test_condition_min_score_not_passed(diabetes, diabetes_model):
    # Arrange
    _, validation = diabetes
    check = PerformanceReport().add_condition_min_score(-100)
    # Act X
    result: List[ConditionResult] = check.conditions_decision(check.run(validation, diabetes_model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details=re.compile('Metric scores that don\'t pass: {\'MSE\':'),
                               name='Metric score is above -100')
    ))


def test_condition_min_score_passed(diabetes, diabetes_model):
    # Arrange
    _, validation = diabetes
    check = PerformanceReport().add_condition_min_score(-5_000)
    # Act X
    result: List[ConditionResult] = check.conditions_decision(check.run(validation, diabetes_model))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Metric score is above -5000')
    ))
