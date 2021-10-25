"""Contains unit tests for the classification_report check."""
from mlchecks.checks.performance import ClassificationReport, classification_report
from mlchecks.utils import MLChecksValueError
from hamcrest import assert_that, calling, raises


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(classification_report).with_args(bad_dataset, None),
                raises(MLChecksValueError,
                       'function classification_report requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(calling(classification_report).with_args(iris_dataset, iris_adaboost),
                raises(MLChecksValueError, 'function classification_report requires dataset to have a label column'))


def test_model_info_object(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = ClassificationReport()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    assert len(result.values()) == 3 # iris has 3 targets
    for col in result.values():
        for val in col.values():
            assert isinstance(val , float)
