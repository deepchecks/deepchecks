import typing as t
import re
from sklearn.ensemble import AdaBoostClassifier
from hamcrest.core.matcher import Matcher
from hamcrest import (
    assert_that, instance_of, has_property, all_of,
    equal_to, has_entries, only_contains, calling, raises
)

from deepchecks import Dataset, CheckResult, ConditionResult, ConditionCategory
from deepchecks.checks import ClassPerformanceImbalanceCheck
from deepchecks.utils import DeepchecksValueError

from tests.checks.utils import equal_condition_result


def test_class_performance_imbalance(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    _, test, model = iris_split_dataset_and_model
    check = ClassPerformanceImbalanceCheck()
    check_result = check.run(dataset=test, model=model)
    validate_class_performance_imbalance_check_result(check_result)


def test_class_performance_imbalance_with_custom_metrics(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    _, test, model = iris_split_dataset_and_model

    alternative_metrics = {
        'Test1': lambda y_true, y_pred: {0: 1., 1: 0.9, 2: 0.89},
        'Test2': lambda y_true, y_pred: {0: 0.9, 1: 0.87, 2: 1.},
        'Test3': lambda y_true, y_pred: {0: 0.97, 1: 1., 2: 0.79}
    }

    check = ClassPerformanceImbalanceCheck(metrics=alternative_metrics)
    check_result = check.run(dataset=test, model=model)

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


def test_class_performance_imbalance_with_custom_metrics_that_return_values_with_incorrect_dtype(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    _, test, model = iris_split_dataset_and_model

    alternative_metrics_1 = {
        'Test1': lambda y_true, y_pred: 1,
    }

    alternative_metrics_2 = {
        'Test3': lambda y_true, y_pred: dict() # dict dtype is allowed but it cannot be empty
    }

    alternative_metrics_3 = {
        'Test3': lambda y_true, y_pred: dict(a="Hello!")  # dict dtype is allowed but values dtype must be int|float
    }

    check1 = ClassPerformanceImbalanceCheck(metrics=alternative_metrics_1) # type: ignore

    assert_that(
        calling(check1.run).with_args(dataset=test, model=model),
        raises(
            DeepchecksValueError,
            r'Check ClassPerformanceImbalanceCheck expecting that alternative metrics will return '
            r"not empty instance of 'Dict\[Hasable, float|int\]', but got tuple"
        )
    )

    check2 = ClassPerformanceImbalanceCheck(metrics=alternative_metrics_2)

    assert_that(
        calling(check2.run).with_args(dataset=test, model=model),
        raises(
            DeepchecksValueError,
            r'Check ClassPerformanceImbalanceCheck expecting that alternative metrics will return '
            r"not empty instance of 'Dict\[Hasable, float\|int\]'"
        )
    )

    check3 = ClassPerformanceImbalanceCheck(metrics=alternative_metrics_3) # type: ignore

    assert_that(
        calling(check3.run).with_args(dataset=test, model=model),
        raises(
            DeepchecksValueError,
            r'Check ClassPerformanceImbalanceCheck expecting that alternative metrics will return '
            r"not empty instance of 'Dict\[Hasable, float\|int\]', but got Dict\[Hashable, str]"
        )
    )


def test_condition_percentage_difference_not_greater_than_threshold_that_should_pass(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    _, test, model = iris_split_dataset_and_model
    check = ClassPerformanceImbalanceCheck().add_condition_percentage_difference_not_greater_than(0.5)
    check_result = check.run(dataset=test, model=model)
    validate_class_performance_imbalance_check_result(check_result)

    condition_result, *_ = check.conditions_decision(check_result)

    assert_that(condition_result, equal_condition_result( # type: ignore
        is_pass=True,
        name=f'Relative percentage difference is not greater than 50.00%',
        details='',
        category=ConditionCategory.FAIL,
    ))


def test_condition_percentage_difference_not_greater_than_threshold_that_should_not_pass(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    _, test, model = iris_split_dataset_and_model
    check = ClassPerformanceImbalanceCheck().add_condition_percentage_difference_not_greater_than(0.2)
    check_result = check.run(dataset=test, model=model)
    validate_class_performance_imbalance_check_result(check_result)

    condition_result, *_ = check.conditions_decision(check_result)

    detail_pattern = re.compile(
        r'Relative percentage difference between highest and lowest classes is greater than 20\.00%:'
        r'(\nMetric: .+, lowest class: \d+, highest class: \d+;)+'
    )

    assert_that(condition_result, equal_condition_result( # type: ignore
        is_pass=False,
        name=f'Relative percentage difference is not greater than 20.00%',
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

