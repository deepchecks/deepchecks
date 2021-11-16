"""Contains unit tests for the performance overfit check."""
import typing as t
from numbers import Number

import pandas as pd

from deepchecks import Dataset
from deepchecks.base.check import Condition
from deepchecks.base.check import ConditionResult
from deepchecks.base.check import ConditionCategory
from deepchecks.checks import TrainTestDifferenceOverfit
from deepchecks.utils import DeepchecksValueError
from deepchecks.metric_utils import DEFAULT_MULTICLASS_METRICS
from hamcrest import assert_that, calling, raises, close_to, starts_with


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(TrainTestDifferenceOverfit().run).with_args(bad_dataset, None, None),
                raises(DeepchecksValueError,
                       'Check TrainTestDifferenceOverfit requires dataset to be of type Dataset. instead '
                       'got: str'))


def test_model_wrong_input(iris_labeled_dataset):
    bad_model = 'wrong_input'
    # Act & Assert
    assert_that(calling(TrainTestDifferenceOverfit().run).with_args(iris_labeled_dataset, iris_labeled_dataset,
                                                                    bad_model),
                raises(DeepchecksValueError,
                       'Model must inherit from one of supported models: .*'))


def test_dataset_no_label(iris_dataset):
    # Assert
    assert_that(calling(TrainTestDifferenceOverfit().run).with_args(iris_dataset, iris_dataset, None),
                raises(DeepchecksValueError, 'Check TrainTestDifferenceOverfit requires dataset to have a '
                                           'label column'))


def test_dataset_no_shared_label(iris_labeled_dataset):
    # Assert
    iris_dataset_2 = Dataset(iris_labeled_dataset.data, label='sepal length (cm)')
    assert_that(calling(TrainTestDifferenceOverfit().run).with_args(iris_labeled_dataset, iris_dataset_2, None),
                raises(DeepchecksValueError,
                       'Check TrainTestDifferenceOverfit requires datasets to share the same label'))


def test_dataset_no_shared_features(iris_labeled_dataset):
    # Assert
    iris_dataset_2 = Dataset(pd.concat(
        [iris_labeled_dataset.data,
         iris_labeled_dataset.data[['sepal length (cm)']].rename(columns={'sepal length (cm)': '1'})],
        axis=1),
        label=iris_labeled_dataset.label_name())
    assert_that(calling(TrainTestDifferenceOverfit().run).with_args(iris_labeled_dataset, iris_dataset_2, None),
                raises(DeepchecksValueError,
                       'Check TrainTestDifferenceOverfit requires datasets to share the same features'))


def test_no_diff(iris_split_dataset_and_model):
    # Arrange
    train, _, model = iris_split_dataset_and_model
    check_obj = TrainTestDifferenceOverfit()
    result = check_obj.run(train, train, model)
    for key, value in result.value.items():
        assert_that(key, any(starts_with(metric_name) for metric_name in DEFAULT_MULTICLASS_METRICS))
        assert_that(value, close_to(0, 0.001))


def test_with_diff(iris_split_dataset_and_model):
    # Arrange
    train, val, model = iris_split_dataset_and_model
    check_obj = TrainTestDifferenceOverfit()
    result = check_obj.run(train, val, model)
    for key, value in result.value.items():
        assert_that(key, any(starts_with(metric_name) for metric_name in DEFAULT_MULTICLASS_METRICS))
        assert_that(value, close_to(-0.035, 0.01))


def test_custom_metrics(iris_split_dataset_and_model):
    # Arrange
    train, val, model = iris_split_dataset_and_model
    check_obj = TrainTestDifferenceOverfit(
        alternative_metrics={'Accuracy': 'accuracy', 'Always 0.5': lambda x, y, z: 0.5}
    )
    result = check_obj.run(train, val, model)
    for key, value in result.value.items():
        assert_that(key, any(starts_with(metric_name) for metric_name in DEFAULT_MULTICLASS_METRICS))
        assert isinstance(value, Number)


def test_train_is_lower_by_condition():
    check = TrainTestDifferenceOverfit()
    
    condition, *_ = t.cast(
        t.List[Condition],
        list(check.add_condition_train_is_lower_by(
            var=0.2,
            metrics=["x1", "x2", "x3"],
            category=ConditionCategory.WARN,
            success_message="cond var:{var}, all metrics:{all_metric_values}",
            failure_message="cond var:{var}, all metrics:{all_metric_values}, failed metrics:{failed_metric_values}",
        )._conditions.values())
    )
    
    condition_satisfying_df = pd.DataFrame.from_dict({
        'Training Metrics': {"x1": 0.88, "x2": 0.64, "x3": 0.71},
        'Test Metrics': {"x1": 0.88, "x2": 0.64, "x3": 0.71},
    })

    result = t.cast(ConditionResult, condition.function(condition_satisfying_df))
    assert result.is_pass is True, result.is_pass
    assert result.category == ConditionCategory.WARN, result.category
    assert "var:0.2" in result.details, result.details
    # assert "all metrics:" in result.details, result.details # TODO

    condition_unsatisfying_df = pd.DataFrame.from_dict({
        'Training Metrics': {"x1": 0.88, "x2": 0.64, "x3": 0.71},
        'Test Metrics': {"x1": 0.5, "x2": 0.2, "x3": 0.3},
    })

    result = t.cast(ConditionResult, condition.function(condition_unsatisfying_df))
    assert result.is_pass is False
    assert result.category == ConditionCategory.WARN, result.category
    assert "var:0.2" in result.details, result.details
    # assert "failed metrics:" in result.details, result.details # TODO

    
def test_train_is_lower_by_condition_for_specified_metric():
    check = TrainTestDifferenceOverfit()

    condition_for_x1_metric, *_ = t.cast(
        t.List[Condition],
        list(check.add_condition_train_is_lower_by(0.2, metrics="x1")._conditions.values())
    )

    check._conditions.clear()

    condition_for_x2_metric, *_ = t.cast(
        t.List[Condition],
        list(check.add_condition_train_is_lower_by(0.2, metrics="x2")._conditions.values())
    )

    check._conditions.clear()

    condition_for_all_metrics, *_ = t.cast(
        t.List[Condition],
        list(check.add_condition_train_is_lower_by(0.2)._conditions.values())
    )

    check._conditions.clear()
    
    df_with_unstatisfying_x1_metric_condition = pd.DataFrame.from_dict({
        'Training Metrics': {"x1": 0.88, "x2": 0.64, "x3": 0.71},
        # x1 metric does not satisfy condition
        'Test Metrics': {"x1": 0.5, "x2": 0.64, "x3": 0.71},
    })

    df_with_unstatisfying_x2_metric_condition = pd.DataFrame.from_dict({
        'Training Metrics': {"x1": 0.88, "x2": 0.64, "x3": 0.71},
        # x2 metric does not satisfy condition
        'Test Metrics': {"x1": 0.79, "x2": 0.17, "x3": 0.71}, 
    })

    result = t.cast(
        ConditionResult, 
        condition_for_x1_metric.function(df_with_unstatisfying_x1_metric_condition)
    )
    assert result.is_pass is False

    result = t.cast(
        ConditionResult, 
        condition_for_x2_metric.function(df_with_unstatisfying_x2_metric_condition)
    )
    assert result.is_pass is False

    result = t.cast(
        ConditionResult, 
        condition_for_x1_metric.function(df_with_unstatisfying_x2_metric_condition)
    )
    assert result.is_pass is True

    result = t.cast(
        ConditionResult, 
        condition_for_x2_metric.function(df_with_unstatisfying_x1_metric_condition)
    )
    assert result.is_pass is True

    result = t.cast(
        ConditionResult, 
        condition_for_all_metrics.function(df_with_unstatisfying_x1_metric_condition)
    )
    assert result.is_pass is False

    result = t.cast(
        ConditionResult, 
        condition_for_all_metrics.function(df_with_unstatisfying_x2_metric_condition)
    )
    assert result.is_pass is False