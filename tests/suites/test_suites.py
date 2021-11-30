"""builtin suites tests"""
#pylint: disable=redefined-outer-name
import typing as t
import pytest
from datetime import datetime

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from hamcrest.core.matcher import Matcher
from hamcrest import assert_that, instance_of, only_contains, any_of

from deepchecks import suites, Dataset, SuiteResult, CheckResult, CheckFailure
from deepchecks.errors import DeepchecksValueError


@pytest.fixture()
def iris(iris_clean) -> t.Tuple[Dataset, Dataset, AdaBoostClassifier]:
    # note: to run classification suite succesfully we need to modify iris dataframe
    # according to suite needs
    df = t.cast(pd.DataFrame, iris_clean.frame)
    df['index'] = range(len(df))
    df['date'] = datetime.now()

    train, test = t.cast(
        t.Tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(df, test_size=0.33, random_state=42)
    )

    train, test = (
        Dataset(train, label='target', date='date', index='index'),
        Dataset(test, label='target', date='date', index='index')
    )

    model = AdaBoostClassifier(random_state=0)
    model.fit(train.features_columns(), train.label_col())

    return train, test, model


def test_classification_suite(iris: t.Tuple[Dataset, Dataset, AdaBoostClassifier]):
    train, test, model = iris
    suite = suites.overall_classification_suite()
    # Have to change min test samples of TrustScoreComparison
    suite[1].min_test_samples = 50
    suite[21].min_test_samples = 50

    arguments = (
        dict(train_dataset=train, test_dataset=test, model=model, check_datasets_policy='both'),
        dict(train_dataset=train, model=model, check_datasets_policy='both'),
        dict(test_dataset=test, model=model, check_datasets_policy='both'),
    )

    for args in arguments:
        result = suite.run(**args)
        validate_suite_result(result, expected_results='mixed')


def test_regression_suite(
    diabetes: t.Tuple[Dataset, Dataset],
    diabetes_model: object
):
    train, test = diabetes
    suite = suites.overall_regression_suite()

    arguments = (
        dict(train_dataset=train, test_dataset=test, model=diabetes_model, check_datasets_policy='both'),
        dict(train_dataset=train, model=diabetes_model, check_datasets_policy='both'),
        dict(test_dataset=test, model=diabetes_model, check_datasets_policy='both'),
    )

    for args in arguments:
        result = suite.run(**args)
        validate_suite_result(result, expected_results='mixed')


def test_generic_suite(
    iris: t.Tuple[Dataset, Dataset, AdaBoostClassifier],
    diabetes_split_dataset_and_model: t.Tuple[Dataset, Dataset, object],
):
    iris_train, iris_test, iris_model = iris
    diabetes_train, diabetes_test, diabetes_model = diabetes_split_dataset_and_model
    suite = suites.overall_generic_suite()

    arguments = (
        dict(train_dataset=iris_train, test_dataset=iris_test, model=iris_model, check_datasets_policy='both'),
        dict(train_dataset=iris_train, model=iris_model, check_datasets_policy='both'),
        dict(test_dataset=iris_test, model=iris_model, check_datasets_policy='both'),
        dict(train_dataset=diabetes_train, model=diabetes_model, check_datasets_policy='both'),
        dict(test_dataset=diabetes_test, model=diabetes_model, check_datasets_policy='both'),
        dict(
            train_dataset=diabetes_train,
            test_dataset=diabetes_test,
            model=diabetes_model,
            check_datasets_policy='both'
        ),
    )

    for args in arguments:
        result = suite.run(**args)
        validate_suite_result(result, expected_results='mixed')


def validate_suite_result(
    result: SuiteResult,
    *,
    expected_results: str = 'only successful',
    exception_matcher: t.Optional[Matcher] = None
):
    """
    Args:
        expected_results (Literal['only successful'] | Literal['only failed'] | Literal['mixed'])
    """
    assert_that(result, instance_of(SuiteResult))
    assert_that(result.results, instance_of(list))

    exception_matcher = exception_matcher or only_contains(instance_of(DeepchecksValueError))

    if expected_results == 'only successful':
        assert_that(result.results, only_contains(any_of( # type: ignore
            instance_of(CheckResult)
        )))

    elif expected_results == 'only failed':
        assert_that(result.results, only_contains(any_of( # type: ignore
            instance_of(CheckFailure)
        )))
        assert_that(
            actual=[it.exception for it in result.results], # type: ignore
            matcher=exception_matcher, # type: ignore
        )

    elif expected_results == 'mixed':
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
