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
from hamcrest import assert_that, instance_of, only_contains, any_of, has_length

from deepchecks import suites, Dataset, SuiteResult, CheckResult, CheckFailure, Suite, SingleDatasetBaseCheck
from deepchecks.errors import DeepchecksBaseError


@pytest.fixture()
def iris(iris_clean) -> t.Tuple[Dataset, Dataset, AdaBoostClassifier]:
    # note: to run classification suite successfully we need to modify iris dataframe
    # according to suite needs
    df = t.cast(pd.DataFrame, iris_clean.frame.copy())
    df['index'] = range(len(df))
    df['date'] = datetime.now()

    train, test = t.cast(
        t.Tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(df, test_size=0.33, random_state=42)
    )

    train, test = (
        Dataset(train, label='target', datetime_name='date', index_name='index'),
        Dataset(test, label='target', datetime_name='date', index_name='index')
    )

    model = AdaBoostClassifier(random_state=0)
    model.fit(train.features_columns, train.label_col)

    return train, test, model


def test_generic_suite(
    iris: t.Tuple[Dataset, Dataset, AdaBoostClassifier],
    diabetes_split_dataset_and_model: t.Tuple[Dataset, Dataset, object],
):
    iris_train, iris_test, iris_model = iris
    diabetes_train, diabetes_test, diabetes_model = diabetes_split_dataset_and_model
    suite = suites.full_suite()

    arguments = (
        dict(train_dataset=iris_train, test_dataset=iris_test, model=iris_model),
        dict(train_dataset=iris_train, model=iris_model),
        dict(test_dataset=iris_test, model=iris_model),
        dict(train_dataset=diabetes_train, model=diabetes_model),
        dict(test_dataset=diabetes_test, model=diabetes_model),
        dict(
            train_dataset=diabetes_train,
            test_dataset=diabetes_test,
            model=diabetes_model
        ),
        dict(model=diabetes_model)
    )

    for args in arguments:
        result = suite.run(**args)
        # Calculate number of expected results
        length = get_expected_results_length(suite, args)
        validate_suite_result(result, length)


def validate_suite_result(
    result: SuiteResult,
    length: int,
    exception_matcher: t.Optional[Matcher] = None
):
    assert_that(result, instance_of(SuiteResult))
    assert_that(result.results, instance_of(list))
    assert_that(result.results, has_length(length))

    exception_matcher = exception_matcher or only_contains(instance_of(DeepchecksBaseError))

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


def get_expected_results_length(suite: Suite, args: t.Dict):
    num_single = len([c for c in suite.checks.values() if isinstance(c, SingleDatasetBaseCheck)])
    num_others = len(suite.checks.values()) - num_single
    multiply = 0
    if 'train_dataset' in args:
        multiply += 1
    if 'test_dataset' in args:
        multiply += 1
    # If no train and no test (only model) there will be single result of check failure
    if multiply == 0:
        multiply = 1

    return num_single * multiply + num_others

