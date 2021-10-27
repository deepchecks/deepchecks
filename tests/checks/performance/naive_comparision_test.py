"""Contains unit tests for the confusion_matrix_report check."""
from mlchecks.checks.performance import naive_comparision, NaiveComparision
from mlchecks.utils import MLChecksValueError
from hamcrest import assert_that, calling, raises, equal_to


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(naive_comparision).with_args(bad_dataset, bad_dataset, None),
                raises(MLChecksValueError,
                       'function naive_comparision requires dataset to be of type Dataset. instead got: str'))


def test_0(iris_split_dataset_and_model):
    train_ds, val_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = NaiveComparision()
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result, equal_to(0))

def test_1(iris_split_dataset_and_model):
    train_ds, val_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = NaiveComparision(native_model_type='statistical')
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result, equal_to(0))

def test_2(iris_split_dataset_and_model):
    train_ds, val_ds, clf = iris_split_dataset_and_model
    # Arrange
    check = NaiveComparision(native_model_type='tree')
    # Act X
    result = check.run(train_ds, val_ds, clf).value
    # Assert
    assert_that(result, equal_to(0))
