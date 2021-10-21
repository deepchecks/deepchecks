"""Contains unit tests for the performance overfit check."""
from numbers import Number

import pandas as pd

from mlchecks import Dataset
from mlchecks.checks import train_validation_difference_overfit, TrainValidationDifferenceOverfit
from mlchecks.utils import MLChecksValueError, DEFAULT_MULTICLASS_METRICS
from hamcrest import assert_that, calling, raises, is_in, close_to, starts_with


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(train_validation_difference_overfit).with_args(bad_dataset, None, None),
                raises(MLChecksValueError,
                       'function train_validation_difference_overfit requires dataset to be of type Dataset. instead '
                       'got: str'))


def test_model_wrong_input(iris_dataset):
    bad_model = 'wrong_input'
    # Act & Assert
    assert_that(calling(train_validation_difference_overfit).with_args(iris_dataset, iris_dataset, bad_model),
                raises(MLChecksValueError,
                       "Model must inherit from one of supported models: .*"))


def test_dataset_no_label(iris_dataset):
    # Assert
    assert_that(calling(train_validation_difference_overfit).with_args(iris_dataset, iris_dataset, None),
                raises(MLChecksValueError, 'function train_validation_difference_overfit requires dataset to have a '
                                           'label column'))


def test_dataset_no_shared_label(iris_labeled_dataset):
    # Assert
    iris_dataset_2 = Dataset(iris_labeled_dataset.data, label='sepal length (cm)')
    assert_that(calling(train_validation_difference_overfit).with_args(iris_labeled_dataset, iris_dataset_2, None),
                raises(MLChecksValueError,
                       'function train_validation_difference_overfit requires datasets to share the same label'))


def test_dataset_no_shared_features(iris_labeled_dataset):
    # Assert
    iris_dataset_2 = Dataset(pd.concat(
        [iris_labeled_dataset.data,
         iris_labeled_dataset.data[['sepal length (cm)']].rename(columns={'sepal length (cm)': '1'})],
        axis=1),
        label=iris_labeled_dataset.label_name())
    assert_that(calling(train_validation_difference_overfit).with_args(iris_labeled_dataset, iris_dataset_2, None),
                raises(MLChecksValueError,
                       'function train_validation_difference_overfit requires datasets to share the same features'))


def test_no_diff(iris_split_dataset_and_model):
    # Arrange
    train, val, model = iris_split_dataset_and_model
    check_obj = TrainValidationDifferenceOverfit()
    result = check_obj.run(train, train, model)
    for key, value in result.value.items():
        assert_that(key, any([starts_with(metric_name) for metric_name in DEFAULT_MULTICLASS_METRICS.keys()]))
        assert_that(value, close_to(0, 0.001))


def test_with_diff(iris_split_dataset_and_model):
    # Arrange
    train, val, model = iris_split_dataset_and_model
    check_obj = TrainValidationDifferenceOverfit()
    result = check_obj.run(train, val, model)
    for key, value in result.value.items():
        assert_that(key, any([starts_with(metric_name) for metric_name in DEFAULT_MULTICLASS_METRICS.keys()]))
        assert_that(value, close_to(-0.035, 0.01))


def test_custom_metrics(iris_split_dataset_and_model):
    # Arrange
    train, val, model = iris_split_dataset_and_model
    check_obj = TrainValidationDifferenceOverfit(
        alternative_metrics={'Accuracy': 'accuracy', 'Always 0.5': lambda x, y, z: 0.5}
    )
    result = check_obj.run(train, val, model)
    for key, value in result.value.items():
        assert_that(key, any([starts_with(metric_name) for metric_name in DEFAULT_MULTICLASS_METRICS.keys()]))
        assert(isinstance(value, Number))
