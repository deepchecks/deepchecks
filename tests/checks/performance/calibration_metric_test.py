"""Contains unit tests for the calibration_metric check."""
from deepchecks.checks.performance import CalibrationMetric
from deepchecks.utils import DeepchecksValueError
from hamcrest import assert_that, calling, raises, has_entries, close_to, has_items

from tests.checks.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(CalibrationMetric().run).with_args(bad_dataset, None),
                raises(DeepchecksValueError,
                       'Check CalibrationMetric requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(calling(CalibrationMetric().run).with_args(iris_dataset, iris_adaboost),
                raises(DeepchecksValueError, 'Check CalibrationMetric requires dataset to have a label column'))


def test_regresion_model(diabetes_split_dataset_and_model):
    # Assert
    train, _, clf = diabetes_split_dataset_and_model
    assert_that(calling(CalibrationMetric().run).with_args(train, clf),
                raises(DeepchecksValueError, r'Check CalibrationMetric Expected model to be a type from'
                                           r' \[\'multiclass\', \'binary\'\], but received model of type: regression'))


def test_model_info_object(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = CalibrationMetric()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert len(result) == 3  # iris has 3 targets

    assert_that(result, has_entries({
        0: close_to(0.99, 0.05),
        1: close_to(0.002, 0.05),
        2: close_to(0.28, 0.05)
    }))


def test_condition(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = CalibrationMetric().add_condition_brier_score_not_greater_than()
    # Act X
    result = check.conditions_decision(check.run(iris_labeled_dataset, iris_adaboost))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Classes with high brier score: {0: \'1\', 2: \'0.28\'}',
                               name='Brier scores are not greater than 0.2')))
