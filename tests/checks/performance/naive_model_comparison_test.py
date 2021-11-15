"""Contains unit tests for the confusion_matrix_report check."""
from deepchecks.checks.performance import NaiveModelComparison
from deepchecks.string_utils import format_number
from deepchecks.utils import DeepchecksValueError
from tests.checks.utils import equal_condition_result

from hamcrest import assert_that, calling, raises, close_to, has_items


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(NaiveModelComparison().run).with_args(bad_dataset, bad_dataset, None),
                raises(DeepchecksValueError,
                       'Check NaiveModelComparison requires dataset to be of type Dataset. instead got: str'))


def test_classification_random(iris_split_dataset_and_model):
    train_ds, val_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = NaiveModelComparison(naive_model_type='random')
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result['given_model_score'], close_to(0.9, 0.5))
    assert_that(result['naive_model_score'], close_to(0.2, 0.5))


def test_classification_statistical(iris_split_dataset_and_model):
    train_ds, val_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = NaiveModelComparison(naive_model_type='statistical')
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result['given_model_score'], close_to(0.9, 0.5))
    assert_that(result['naive_model_score'], close_to(0.3, 0.5))


def test_classification_tree_custom_metric(iris_split_dataset_and_model):
    train_ds, val_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = NaiveModelComparison(naive_model_type='random', metric='recall_micro')
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result['given_model_score'], close_to(0.9, 0.5))
    assert_that(result['naive_model_score'], close_to(0.2, 0.5))


def test_regression_random(diabetes_split_dataset_and_model):
    train_ds, val_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = NaiveModelComparison(naive_model_type='random')
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result['given_model_score'], close_to(-57, 0.5))
    assert_that(result['naive_model_score'], close_to(-105, 0.5))


def test_regression_statistical(diabetes_split_dataset_and_model):
    train_ds, val_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = NaiveModelComparison(naive_model_type='statistical')
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result['given_model_score'], close_to(-57, 0.5))
    assert_that(result['naive_model_score'], close_to(-76, 0.5))


def test_condition_ratio_more_than_not_passed(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, val_ds, clf = diabetes_split_dataset_and_model
    check = NaiveModelComparison().add_condition_effective_ratio_more_than(min_allowed_effective_ratio=1.4)

    # Act
    check_result = check.run(train_ds, val_ds, clf)
    condition_result = check.conditions_decision(check_result)
    effective_ratio = check_result.value['effective_ratio']

    assert_that(effective_ratio, close_to(1.32, 0.03))
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               name='More than 1.4 effective ratio '
                                    'between the checked model\'s result and the naive model\'s result',
                               details=f'The checked model is {format_number(effective_ratio)} times as effective as' \
                                       ' the naive model using the given metric')
    ))


def test_condition_ratio_more_than_passed(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, val_ds, clf = diabetes_split_dataset_and_model
    check = NaiveModelComparison().add_condition_effective_ratio_more_than(min_allowed_effective_ratio=1.3)

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='More than 1.3 effective ratio '
                                    'between the checked model\'s result and the naive model\'s result')
    ))
