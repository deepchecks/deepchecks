import typing as t
import pandas as pd
from hamcrest import assert_that, instance_of, has_items, only_contains, any_of

from deepchecks import suites, Dataset, SuiteResult, CheckResult, CheckFailure
from deepchecks.utils import DeepchecksValueError


def test_classification_suite(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]
):
    train, test, model = iris_split_dataset_and_model
    arguments = (
        dict(train_dataset=train, test_dataset=test, model=model, check_datasets_policy='both'),
        dict(train_dataset=train, model=model, check_datasets_policy='both'),
        dict(test_dataset=test, model=model, check_datasets_policy='both'),
    )

    for args in arguments:
        result = suites.OverallClassificationCheckSuite.run(**args)
        assert_that(result, instance_of(SuiteResult))
        assert_that(result.results, instance_of(list))
        if len(result.results) != 0:
            assert_that(result.results, only_contains(instance_of(CheckResult))) # type: ignore

def test_classification_suite_with_empty_datasets(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]
):
    train, test, model = iris_split_dataset_and_model
    empty_train = Dataset(pd.DataFrame())
    empty_test = Dataset(pd.DataFrame())

    arguments = (
        dict(train_dataset=empty_train, test_dataset=test, check_datasets_policy='both'),
        dict(train_dataset=train, test_dataset=empty_test, check_datasets_policy='both'),
        dict(train_dataset=empty_train, test_dataset=empty_test, check_datasets_policy='both')
    )
    
    for args in arguments:
        result = suites.OverallClassificationCheckSuite.run(**args)
        assert_that(result, instance_of(SuiteResult))
        assert_that(result.results, instance_of(list))
        assert_that(result.results, only_contains(any_of( # type: ignore
            instance_of(CheckFailure),
            instance_of(CheckResult),
        )))

        failures = [it.exception for it in result.results if isinstance(it, CheckFailure)]
        assert_that(len(failures) != 0)
        assert_that(failures, only_contains(instance_of(DeepchecksValueError))) # type: ignore


def test_classification_suite_with_wrong_model_type(
    diabetes: t.Tuple[Dataset, Dataset],
    diabetes_model: object
):
    train, test = diabetes
    result = suites.OverallClassificationCheckSuite.run(train_dataset=train,test_dataset=test)
    
    assert_that(result, instance_of(SuiteResult))
    assert_that(result.results, instance_of(list))
    assert_that(result.results, only_contains(any_of( # type: ignore
        instance_of(CheckFailure),
        instance_of(CheckResult),
    )))

    failures = [it.exception for it in result.results if isinstance(it, CheckFailure)]
    assert_that(len(failures) != 0)
    assert_that(failures, only_contains(instance_of(DeepchecksValueError))) # type: ignore


def test_regression_suite(
    diabetes: t.Tuple[Dataset, Dataset],
    diabetes_model: object
):
    train, test = diabetes

    arguments = (
        dict(train_dataset=train, test_dataset=test, model=diabetes_model),
        dict(train_dataset=train, model=diabetes_model),
        dict(test_dataset=test, model=diabetes_model),
    )

    for args in arguments:
        result = suites.OverallRegressionCheckSuite.run(**args)
        assert_that(result, instance_of(SuiteResult))
        assert_that(result.results, instance_of(list))
        if len(result.results) != 0:
            assert_that(actual=result.results, matcher=only_contains(instance_of(CheckResult))) # type: ignore


def test_regression_suite_with_empty_datasets(
    diabetes: t.Tuple[Dataset, Dataset],
    diabetes_model: object
):
    train, test = diabetes
    empty_train = Dataset(pd.DataFrame())
    empty_test = Dataset(pd.DataFrame())

    arguments = (
        dict(train_dataset=empty_train, test_dataset=test, model=diabetes_model),
        dict(train_dataset=train, test_dataset=empty_test, model=diabetes_model),
        dict(train_dataset=empty_train, test_dataset=empty_test, model=diabetes_model)
    )

    for args in arguments:
        result = suites.OverallRegressionCheckSuite.run(**args)
        assert_that(result, instance_of(SuiteResult))
        assert_that(result.results, instance_of(list))
        assert_that(result.results, only_contains(any_of( # type: ignore
            instance_of(CheckFailure),
            instance_of(CheckResult),
        )))

        failures = [it.exception for it in result.results if isinstance(it, CheckFailure)]
        assert_that(len(failures) != 0)
        assert_that(failures, only_contains(instance_of(DeepchecksValueError))) # type: ignore


def test_regression_suite_with_wrong_model_type(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]
):
    train, test, model = iris_split_dataset_and_model
    result = suites.OverallRegressionCheckSuite.run(train_dataset=train,test_dataset=test)
    
    assert_that(result, instance_of(SuiteResult))
    assert_that(result.results, instance_of(list))
    assert_that(result.results, only_contains(any_of( # type: ignore
        instance_of(CheckFailure),
        instance_of(CheckResult),
    )))

    failures = [it.exception for it in result.results if isinstance(it, CheckFailure)]
    assert_that(len(failures) != 0)
    assert_that(failures, only_contains(instance_of(DeepchecksValueError))) # type: ignore


def test_generic_suite(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object],
    diabetes_split_dataset_and_model: t.Tuple[Dataset, Dataset, object],
):
    iris_train, iris_test, iris_model = iris_split_dataset_and_model
    diabetes_train, diabetes_test, diabetes_model = diabetes_split_dataset_and_model

    arguments = (
        dict(train_dataset=iris_train, test_dataset=iris_test, model=iris_model),
        dict(train_dataset=iris_train, model=iris_model),
        dict(test_dataset=iris_test, model=iris_model),
        dict(train_dataset=diabetes_train, test_dataset=diabetes_test, model=diabetes_model),
        dict(train_dataset=diabetes_train, model=diabetes_model),
        dict(test_dataset=diabetes_test, model=diabetes_model),
    )

    for args in arguments:
        result = suites.OverallGenericCheckSuite.run(**args)
        assert_that(result, instance_of(SuiteResult))
        assert_that(result.results, instance_of(list))
        if len(result.results) != 0:
            assert_that(result.results, only_contains(instance_of(CheckResult))) # type: ignore


def test_suites_run_with_datasets_that_has_different_features(
    iris_split_dataset_and_model: t.Tuple[Dataset, Dataset, object],
    diabetes_split_dataset_and_model: t.Tuple[Dataset, Dataset, object]
):
    iris_train, _, _ = iris_split_dataset_and_model
    _, diabetes_test, _ = diabetes_split_dataset_and_model

    run_methods = (
        suites.OverallClassificationCheckSuite.run,
        suites.OverallRegressionCheckSuite.run,
        suites.OverallGenericCheckSuite.run,
    )

    for run_method in run_methods:
        result = run_method(train_dataset=iris_train, test_dataset=diabetes_test)
        assert_that(result, instance_of(SuiteResult))
        assert_that(result.results, instance_of(list))
        assert_that(result.results, only_contains(any_of( # type: ignore
            instance_of(CheckFailure), instance_of(CheckResult),
        )))

        failures = [it.exception for it in result.results if isinstance(it, CheckFailure)]
        assert_that(len(failures) != 0)
        assert_that(failures, only_contains(instance_of(DeepchecksValueError))) # type: ignore