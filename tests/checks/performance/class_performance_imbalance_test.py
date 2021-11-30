import typing as t
import re
from sklearn.ensemble import AdaBoostClassifier
from hamcrest.core.matcher import Matcher
from hamcrest import (
    assert_that, instance_of, has_property, all_of,
    equal_to, has_entries, only_contains, calling, raises
)

from deepchecks import Dataset, CheckResult, ConditionCategory
from deepchecks.checks import ClassPerformanceImbalanceCheck
from deepchecks.errors import DeepchecksValueError

from tests.checks.utils import equal_condition_result


def test_class_performance_imbalance(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = ClassPerformanceImbalanceCheck()
    # Act
    check_result = check.run(dataset=test, model=model)
    # Assert
    validate_class_performance_imbalance_check_result(check_result)


def test_init_class_performance_imbalance_with_empty_dict_of_metrics():
    assert_that(
        calling(ClassPerformanceImbalanceCheck).with_args(alternative_metrics=dict()),
        raises(DeepchecksValueError, 'alternative_metrics - expected to receive not empty dict of scorers!')
    )


def test_init_class_performance_imbalance_with_metrics_dict_that_contains_not_callable_and_not_name_of_sklearn_scorer():
    assert_that(
        calling(ClassPerformanceImbalanceCheck).with_args(alternative_metrics=dict(Metric=1)),
        raises(
            DeepchecksValueError,
            r"alternative_metrics - expected to receive 'Mapping\[str, Callable\]' but got 'Mapping\[str, int\]'!"
        )
    )


def test_class_performance_imbalance_with_custom_metrics(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    alternative_metrics = {
        'Test1': lambda model, features, labels: {0: 1., 1: 0.9, 2: 0.89},
        'Test2': lambda model, features, labels: {0: 0.9, 1: 0.87, 2: 1.},
        'Test3': lambda model, features, labels: {0: 0.97, 1: 1., 2: 0.79}
    }
    check = ClassPerformanceImbalanceCheck(alternative_metrics=alternative_metrics)

    # Act
    check_result = check.run(dataset=test, model=model)
    
    # Assert
    class_metrics_matcher = has_entries({
        k: instance_of(float)
        for k in alternative_metrics.keys()
    })
    value_matcher = all_of(
        instance_of(dict),
        has_entries({
            0: all_of(instance_of(dict), class_metrics_matcher),
            1: all_of(instance_of(dict), class_metrics_matcher),
            2: all_of(instance_of(dict), class_metrics_matcher),
        })
    )

    validate_class_performance_imbalance_check_result(
        check_result,
        value=value_matcher
    )


def test_class_performance_imbalance_with_custom_metrics_that_return_not_dict_value(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model

    alternative_metrics = {'Test1': lambda model, features, labels: 1,}
    check = ClassPerformanceImbalanceCheck(alternative_metrics=alternative_metrics) # type: ignore

    # Assert
    assert_that(
        calling(check.run).with_args(dataset=test, model=model),
        raises(
            DeepchecksValueError,
            r'Check ClassPerformanceImbalanceCheck expecting that alternative metrics will return '
            r"not empty instance of 'Mapping\[Hashable, float|int\]', but got tuple"
        )
    )


def test_class_performance_imbalance_with_custom_metrics_that_return_empty_dict(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrage
    _, test, model = iris_split_dataset_and_model

    # dict dtype is allowed but it cannot be empty
    alternative_metrics = {'Test3': lambda model, features, labels: dict()}
    check = ClassPerformanceImbalanceCheck(alternative_metrics=alternative_metrics)

    # Assert
    assert_that(
        calling(check.run).with_args(dataset=test, model=model),
        raises(
            DeepchecksValueError,
            r'Check ClassPerformanceImbalanceCheck expecting that alternative metrics will return '
            r"not empty instance of 'Mapping\[Hashable, float\|int\]'"
        )
    )


def test_class_performance_imbalance_with_custom_metrics_that_return_values_with_incorrect_dtype(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model

    # dict dtype is allowed but values dtype must be int|float
    alternative_metrics = {'Test3': lambda model, features, labels: dict(a="Hello!")}
    check = ClassPerformanceImbalanceCheck(alternative_metrics=alternative_metrics)

    # Assert
    assert_that(
        calling(check.run).with_args(dataset=test, model=model),
        raises(
            DeepchecksValueError,
            r'Check ClassPerformanceImbalanceCheck expecting that alternative metrics will return '
            r"not empty instance of 'Mapping\[Hashable, float\|int\]', but got 'Mapping\[Hashable, str]'"
        )
    )


def test_condition_percentage_difference_not_greater_than_threshold_that_should_pass(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = ClassPerformanceImbalanceCheck().add_condition_ratio_difference_not_greater_than(0.5)

    # Act
    check_result = check.run(dataset=test, model=model)
    validate_class_performance_imbalance_check_result(check_result)

    condition_result, *_ = check.conditions_decision(check_result)

    # Assert
    assert_that(condition_result, equal_condition_result( # type: ignore
        is_pass=True,
        name=f'Relative ratio difference is not greater than 50.00%',
        details='',
        category=ConditionCategory.FAIL,
    ))


def test_condition_percentage_difference_not_greater_than_threshold_that_should_not_pass(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    # Arrange
    _, test, model = iris_split_dataset_and_model
    check = ClassPerformanceImbalanceCheck().add_condition_ratio_difference_not_greater_than(0.2)

    # Act
    check_result = check.run(dataset=test, model=model)
    validate_class_performance_imbalance_check_result(check_result)
    condition_result, *_ = check.conditions_decision(check_result)

    # Assert
    detail_pattern = re.compile(
        r'Relative ratio difference between highest and lowest classes is greater than 20\.00%:'
        r'(\nMetric: .+, lowest class: \d+, highest class: \d+;)+'
    )
    assert_that(condition_result, equal_condition_result( # type: ignore
        is_pass=False,
        name=f'Relative ratio difference is not greater than 20.00%',
        details=detail_pattern,
        category=ConditionCategory.FAIL,
    ))


def validate_class_performance_imbalance_check_result(
    result: CheckResult,
    value: t.Optional[Matcher] = None
):
    if value is None:
        default_metrics_matcher = has_entries({
            'Accuracy': instance_of(float),
            'Precision': instance_of(float),
            'Recall': instance_of(float),
        })

        value_matcher = all_of(
            instance_of(dict),
            has_entries({
                0: all_of(instance_of(dict), default_metrics_matcher),
                1: all_of(instance_of(dict), default_metrics_matcher),
                2: all_of(instance_of(dict), default_metrics_matcher),
            })
        )
    else:
        value_matcher = value

    assert_that(result, all_of(
        instance_of(CheckResult),
        has_property('value', value_matcher),
        has_property('header', all_of(instance_of(str), equal_to('Class Performance Imbalance'))),
        has_property('display', only_contains(has_property('__call__')))
    ))

