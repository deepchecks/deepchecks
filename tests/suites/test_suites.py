"""builtin suites tests"""
import typing as t
from sklearn.ensemble import AdaBoostClassifier
from hamcrest.core.matcher import Matcher
from hamcrest import assert_that, instance_of, only_contains, any_of

from deepchecks import suites, Dataset, SuiteResult, CheckResult, CheckFailure
from deepchecks.utils import DeepchecksValueError


def test_classification_suite(iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]):
    train, test, model = iris_split_dataset_and_model
    suite = suites.overall_classification_check_suite()

    arguments = (
        dict(train_dataset=train, test_dataset=test, model=model, check_datasets_policy='both'),
        dict(train_dataset=train, model=model, check_datasets_policy='both'),
        dict(test_dataset=test, model=model, check_datasets_policy='both'),
    )

    for args in arguments:
        result = suite.run(**args)
        validate_suite_result(result)


def test_regression_suite(
    diabetes: t.Tuple[Dataset, Dataset],
    diabetes_model: object
):
    train, test = diabetes
    suite = suites.overall_regression_check_suite()

    arguments = (
        dict(train_dataset=train, test_dataset=test, model=diabetes_model, check_datasets_policy='both'),
        dict(train_dataset=train, model=diabetes_model, check_datasets_policy='both'),
        dict(test_dataset=test, model=diabetes_model, check_datasets_policy='both'),
    )

    for args in arguments:
        result = suite.run(**args)
        validate_suite_result(result)


def test_generic_suite(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, AdaBoostClassifier],
    diabetes_split_dataset_and_model: t.Tuple[Dataset, Dataset, object],
):
    iris_train, iris_test, iris_model = iris_split_dataset_and_model
    diabetes_train, diabetes_test, diabetes_model = diabetes_split_dataset_and_model
    suite = suites.overall_generic_check_suite()

    arguments = (
        dict(train_dataset=iris_train, test_dataset=iris_test, model=iris_model, check_datasets_policy='both'),
        dict(train_dataset=iris_train, model=iris_model, check_datasets_policy='both'),
        dict(test_dataset=iris_test, model=iris_model, check_datasets_policy='both'),
        dict(train_dataset=diabetes_train, test_dataset=diabetes_test, model=diabetes_model, check_datasets_policy='both'),
        dict(train_dataset=diabetes_train, model=diabetes_model, check_datasets_policy='both'),
        dict(test_dataset=diabetes_test, model=diabetes_model, check_datasets_policy='both'),
    )

    for args in arguments:
        result = suite.run(**args)
        validate_suite_result(result)


def validate_suite_result(
    result: SuiteResult,
    *,
    expected_results: str = "only successful",
    exception_matcher: t.Optional[Matcher] = None
):
    """
    Args:
        expected_results (Literal['only successful'] | Literal['only failed'] | Literal['mixed'])
    """
    assert_that(result, instance_of(SuiteResult))
    assert_that(result.results, instance_of(list))

    exception_matcher = exception_matcher or only_contains(instance_of(DeepchecksValueError))

    if expected_results == "only successful":
        assert_that(result.results, only_contains(any_of( # type: ignore
            instance_of(CheckResult)
        )))
    
    elif expected_results == "only failed":
        assert_that(result.results, only_contains(any_of( # type: ignore
            instance_of(CheckFailure)
        )))
        assert_that(
            actual=[it.exception for it in result.results], # type: ignore
            matcher=exception_matcher, # type: ignore
        )
    
    elif expected_results == "mixed":
        assert_that(result.results, only_contains(any_of( # type: ignore
            instance_of(CheckFailure),
            instance_of(CheckResult),
        )))
        
        failures = [
            it.exception 
            for it in result.results 
            if isinstance(it, CheckFailure)
        ]
        
        if len(failures) != 0:
            assert_that(actual=failures, matcher=exception_matcher) # type: ignore
    
    else:
        raise ValueError(f'Unknown value of "expected_results" - {expected_results}')