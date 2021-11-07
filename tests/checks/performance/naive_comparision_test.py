"""Contains unit tests for the confusion_matrix_report check."""
from mlchecks.checks.performance import NaiveModelComparison
from mlchecks.utils import MLChecksValueError
from hamcrest import assert_that, calling, raises, close_to


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(NaiveModelComparison().run).with_args(bad_dataset, bad_dataset, None),
                raises(MLChecksValueError,
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
    assert_that(result['given_model_score'], close_to(57, 0.5))
    assert_that(result['naive_model_score'], close_to(105, 0.5))


def test_regression_statistical(diabetes_split_dataset_and_model):
    train_ds, val_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = NaiveModelComparison(naive_model_type='statistical')
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result['given_model_score'], close_to(57, 0.5))
    assert_that(result['naive_model_score'], close_to(76, 0.5))
