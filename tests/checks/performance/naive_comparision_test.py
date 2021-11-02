"""Contains unit tests for the confusion_matrix_report check."""
from mlchecks.checks.performance import naive_comparision, NaiveComparision
from mlchecks.utils import MLChecksValueError
from hamcrest import assert_that, calling, raises, close_to


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(naive_comparision).with_args(bad_dataset, bad_dataset, None),
                raises(MLChecksValueError,
                       'function naive_comparision requires dataset to be of type Dataset. instead got: str'))


def test_classification_random(iris_split_dataset_and_model):
    train_ds, val_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = NaiveComparision()
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result, close_to(4.6, 0.05))

def test_classification_statistical(iris_split_dataset_and_model):
    train_ds, val_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = NaiveComparision(native_model_type='statistical')
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result, close_to(3, 0.5))

def test_classification_tree(iris_split_dataset_and_model):
    train_ds, val_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = NaiveComparision(native_model_type='tree')
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result, close_to(0.95, 0.05))

def test_classification_tree_custom_metric(iris_split_dataset_and_model):
    train_ds, val_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = NaiveComparision(native_model_type='tree', metric='recall_micro')
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result, close_to(0.95, 0.05))

def test_regression_random(diabetes_split_dataset_and_model):
    train_ds, val_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = NaiveComparision()
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result, close_to(0.5, 0.05))

def test_regression_statistical(diabetes_split_dataset_and_model):
    train_ds, val_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = NaiveComparision(native_model_type='statistical')
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result, close_to(0.7, 0.5))

def test_regression_tree(diabetes_split_dataset_and_model):
    train_ds, val_ds, clf = diabetes_split_dataset_and_model
    # Arrange
    check = NaiveComparision(native_model_type='tree')
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result, close_to(0.7, 0.05))
