# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
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
    df = t.cast(pd.DataFrame, iris_clean.frame.copy())
    df['index'] = range(len(df))
    df['date'] = datetime.now()

    train, test = t.cast(
        t.Tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(df, test_size=0.33, random_state=42)
    )

    train, test = (
        Dataset(train, label_name='target', date_name='date', index_name='index'),
        Dataset(test, label_name='target', date_name='date', index_name='index')
    )

    model = AdaBoostClassifier(random_state=0)
    model.fit(train.features_columns, train.label_col)

    return train, test, model


@pytest.fixture()
def iris_with_non_textual_columns(iris_clean) -> t.Tuple[Dataset, Dataset, AdaBoostClassifier]:
    df = t.cast(pd.DataFrame, iris_clean.frame.copy())

    # TODO: generate non textual columns automaticly, use not only integer
    df[5] = range(len(df))
    df[6] = datetime.now()

    # NOTE:
    # if you try to use some random integers as column names
    # then with big probability test will fall
    #
    # it looks like sklearn requires column names with dtype int to be in range [0, n_columns]
    #
    # in my case UnusedFeatures check was the reason why test failed
    # more precisly it failed at the next line in unused_features.py module:

    # >>> ... pre_pca_transformer.fit_transform(
    # ...        dataset.features_columns().sample(n_samples, random_state=self.random_state)
    # ...    ) ...
    #
    # with next exception: ValueError('all features must be in [0, 3] or [-4, 0]')

    renamer = {
        'sepal length (cm)': 0,
        'sepal width (cm)': 1,
        'petal length (cm)': 2,
        'petal width (cm)': 3,
        'target': 4
    }

    train, test = t.cast(
        t.Tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(df, test_size=0.33, random_state=42)
    )

    train, test = (
        Dataset(
            train.rename(columns=renamer),
            features=list(renamer.values())[:-1],
            label_name=4, date_name=6, index_name=5
        ),
        Dataset(
            test.rename(columns=renamer),
            features=list(renamer.values())[:-1],
            label_name=4, date_name=6, index_name=5
        )
    )

    model = AdaBoostClassifier(random_state=0)
    model.fit(train.features_columns, train.label_col)

    return train, test, model


def test_classification_suite(iris: t.Tuple[Dataset, Dataset, AdaBoostClassifier]):
    train, test, model = iris
    suite = suites.overall_classification_suite()

    # TODO: Have to change min test samples of TrustScoreComparison
    # suite[1].min_test_samples = 50
    # suite[21].min_test_samples = 50

    arguments = (
        dict(train_dataset=train, test_dataset=test, model=model, check_datasets_policy='both'),
        dict(train_dataset=train, model=model, check_datasets_policy='both'),
        dict(test_dataset=test, model=model, check_datasets_policy='both'),
    )

    for args in arguments:
        result = suite.run(**args)
        validate_suite_result(result, expected_results='mixed')


def test_overall_suite_with_datasets_that_have_non_textual_columns(
    iris_with_non_textual_columns: t.Tuple[Dataset, Dataset, AdaBoostClassifier]
):
    train, test, model = iris_with_non_textual_columns
    suite = suites.overall_suite()

    # # TODO: Have to change min test samples of TrustScoreComparison
    # suite[1].min_test_samples = 50

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
