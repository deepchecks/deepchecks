"""Contains unit tests for the confusion_matrix_report check."""
from deepchecks.checks.performance import NaiveModelComparison
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


def test_condition_max_ratio_not_passed(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, val_ds, clf = diabetes_split_dataset_and_model
    check = NaiveModelComparison().add_condition_max_effective_ratio()

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='Not more than 0.7 effective ratio '
                                    'between the naive model\'s result and the checked model\'s result',
                               details='The naive model is 0.76 times as effective as' \
                                       ' the checked model using the given metric, which is more than the'
                                       ' allowed ratio of: 0.7')
    ))


def test_condition_max_ratio_passed(diabetes_split_dataset_and_model):
    # Arrange
    train_ds, val_ds, clf = diabetes_split_dataset_and_model
    check = NaiveModelComparison().add_condition_max_effective_ratio(max_allowed_effective_ratio=0.8)

    # Act
    result = check.conditions_decision(check.run(train_ds, val_ds, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Not more than 0.8 effective ratio '
                                    'between the naive model\'s result and the checked model\'s result')
    ))

