import typing as t
import re
from hamcrest import assert_that, instance_of, only_contains, matches_regexp

from deepchecks import Dataset, CheckResult, ConditionCategory
from deepchecks.checks.performance import ModelInferenceTimeCheck

from tests.checks.utils import equal_condition_result, SCIENTIFIC_NOTATION_REGEXP


def test_model_inference_time_check(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]
):
    _, test, model = iris_split_dataset_and_model
    check = ModelInferenceTimeCheck()

    result = check.run(test, model)

    assert_that(result, instance_of(CheckResult))
    assert_that(result.value, instance_of(float))
    assert_that(result.display, instance_of(list))
    assert_that(result.display, only_contains(instance_of(str))) # type: ignore

    details_pattern = (
        r'Average model inference time for one sample \(in seconds\) '
        fr'equal to {SCIENTIFIC_NOTATION_REGEXP.pattern}'
    )

    assert_that(result.display[0], matches_regexp(details_pattern)) # type: ignore


def test_model_inference_time_check_with_condition_that_should_pass(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]
):
    _, test, model = iris_split_dataset_and_model
    check = ModelInferenceTimeCheck().add_condition_inference_time_is_not_greater_than(0.1)

    result = check.run(test, model)
    condition_result, *_ = check.conditions_decision(result)

    name = (
        'Average model inference time for one sample is not '
        'greater than 0.1'
    )

    assert_that(condition_result, equal_condition_result( # type: ignore
        is_pass=True,
        category=ConditionCategory.FAIL,
        name=name,
        details=''
    ))


def test_model_inference_time_check_with_condition_that_should_not_pass(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]
):
    _, test, model = iris_split_dataset_and_model
    check = ModelInferenceTimeCheck().add_condition_inference_time_is_not_greater_than(0.00000001)

    result = check.run(test, model)
    condition_result, *_ = check.conditions_decision(result)

    name = (
        'Average model inference time for one sample is not '
        'greater than 1e-08'
    )
    details_pattern = re.compile(
        r'Average model inference time for one sample \(in seconds\) '
        fr'is greater than {SCIENTIFIC_NOTATION_REGEXP.pattern}'
    )

    assert_that(condition_result, equal_condition_result( # type: ignore
        is_pass=False,
        category=ConditionCategory.FAIL,
        name=name,
        details=details_pattern
    ))
