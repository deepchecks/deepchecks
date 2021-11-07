"""Contains unit tests for the performance report check."""
from mlchecks.checks.performance import PerformanceReport
from mlchecks.utils import MLChecksValueError
from hamcrest import assert_that, calling, raises, close_to, has_entries


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(PerformanceReport().run).with_args(bad_dataset, None),
                raises(MLChecksValueError,
                       'function _performance_report requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(calling(PerformanceReport().run).with_args(iris_dataset, iris_adaboost),
                raises(MLChecksValueError, 'function _performance_report requires dataset to have a label column'))


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
        'RMSE': close_to(50, 20),
        'MSE': close_to(3200, 1000),
    }))
