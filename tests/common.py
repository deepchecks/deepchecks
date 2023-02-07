# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
# pylint: disable=unused-wildcard-import, wildcard-import
"""Common functions."""
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from hamcrest import *
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.matcher import Matcher
from plotly.graph_objects import Histogram

from deepchecks.core import BaseSuite
from deepchecks.core.check_result import CheckFailure, CheckResult, DisplayMap
from deepchecks.core.checks import BaseCheck, SingleDatasetBaseCheck
from deepchecks.core.condition import ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksBaseError
from deepchecks.core.serialization.abc import (HTMLFormatter, IPythonDisplayFormatter, JPEGFormatter, JSONFormatter,
                                               MarkdownFormatter, MimeBundleFormatter, PNGFormatter)
from deepchecks.core.suite import SuiteResult


class DummyCheck(BaseCheck):
    """Dummy check type for testing purpose."""

    def run(self, *args, **kwargs):
        raise NotImplementedError()


def create_suite_result(
    name: str = 'Dummy Suite Result',
    n_of_results: int = 5,  # for each category
    n_of_failures: int = 5,
    include_results_without_conditions: bool = True,
    include_results_without_display: bool = True,
    include_results_without_conditions_and_display: bool = True,
) -> SuiteResult:
    results = [
        create_check_result(value=i, header=f'Dummy Result {i}')
        for i in range(n_of_results)
    ]
    if include_results_without_conditions:
        results.extend([
            create_check_result(
                value=i,
                header=f'Dummy Result Without Conditions {i}',
                include_conditions=False,
            )
            for i in range(n_of_results)
        ])
    if include_results_without_display:
        results.extend([
            create_check_result(
                value=i,
                header=f'Dummy Result Without Display {i}',
                include_display=False,
            )
            for i in range(n_of_results)
        ])
    if include_results_without_conditions_and_display:
        results.extend([
            create_check_result(
                value=i,
                header=f'Dummy Result Without Display and Conditions {i}',
                include_display=False,
                include_conditions=False,
            )
            for i in range(n_of_results)
        ])
    failures = [
        CheckFailure(DummyCheck(), Exception(f'Exception Message {i}'))
        for i in range(n_of_failures)
    ]
    return SuiteResult(
        name=name,
        results=[*results, *failures],
        extra_info=['Some extra info regarding suite']
    )


def create_check_result(
    value: t.Any = None,
    header: str = 'Dummy Result',
    include_display: bool = True,
    include_conditions: bool = True
) -> CheckResult:
    display = (
        [*create_check_result_display(), DisplayMap(a=create_check_result_display())]
        if include_display
        else None
    )
    result = CheckResult(
        value=value or 1000,
        header=header,
        display=display,
    )

    if include_conditions:
        c1 = ConditionResult(ConditionCategory.WARN, 'Dummy Condition 1')
        c2 = ConditionResult(ConditionCategory.FAIL, 'Dummy Condition 2')
        c3 = ConditionResult(ConditionCategory.PASS, 'Dummy Condition 3')
        c1.set_name('Dummy Condition 1')
        c2.set_name('Dummy Condition 3')
        c3.set_name('Dummy Condition 3')
        result.conditions_results = [c1, c2, c3]

    result.check = DummyCheck()
    return result


def create_check_result_display():
    def draw_plot():
        plt.subplots()
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])

    return [
        'Hello world',
        pd.DataFrame({'foo': range(3), 'bar': range(3)}),
        pd.DataFrame({'foo': range(3), 'bar': range(3)}).style,
        px.bar(
            px.data.gapminder().query("country == 'Canada'"),
            x='year', y='pop'
        ),
        draw_plot
    ]


def instance_of_ipython_formatter():
    return any_of(
        instance_of(HTMLFormatter),
        instance_of(MarkdownFormatter),
        instance_of(PNGFormatter),
        instance_of(JPEGFormatter),
        instance_of(JSONFormatter),
        instance_of(IPythonDisplayFormatter),
        instance_of(MimeBundleFormatter),
    )


def assert_class_performance_display(
    xaxis,  # classes
    yaxis,  # values
    metrics=('Recall', 'Precision'),
):
    pairs = (
        (dataset, metric)
        for dataset in ('Train', 'Test')
        for metric in metrics
    )
    return contains_exactly(*[
        all_of(
            instance_of(Histogram),
            has_property('name', equal_to(dataset)),
            has_property('hovertemplate', contains_string(f'Metric={metric}')),
            has_property('x', xaxis[index]),
            has_property('y', yaxis[index]),
        )
        for index, (dataset, metric) in enumerate(pairs)
    ])


class IsNan(BaseMatcher):

    def _matches(self, item):
        return np.isnan(item)

    def describe_to(self, description):
        description.append_text('item is not nan')


def is_nan():
    return IsNan()


def get_expected_results_length(
    suite: BaseSuite,
    args: t.Dict[t.Any, t.Any]
):
    num_single = len([c for c in suite.checks.values() if isinstance(c, SingleDatasetBaseCheck)])
    num_others = len(suite.checks.values()) - num_single
    multiply = 0

    if 'train_dataset' in args and args['train_dataset'] is not None:
        multiply += 1
    if 'test_dataset' in args and args['test_dataset'] is not None:
        multiply += 1
    # If no train and no test (only model) there will be single result of check failure
    if multiply == 0:
        multiply = 1

    return num_single * multiply + num_others


def validate_suite_result(
    result: SuiteResult,
    min_length: int,
    exception_matcher: t.Optional[Matcher] = None
):
    assert_that(result, instance_of(SuiteResult))
    assert_that(result.results, instance_of(list))
    assert_that(len(result.results) >= min_length)

    exception_matcher = exception_matcher or only_contains(instance_of(DeepchecksBaseError))

    assert_that(result.results, only_contains(any_of(  # type: ignore
        instance_of(CheckFailure),
        instance_of(CheckResult),
    )))

    failures = [
        it.exception
        for it in result.results
        if isinstance(it, CheckFailure)
    ]

    if len(failures) != 0:
        assert_that(failures, matcher=exception_matcher)  # type: ignore

    for check_result in result.results:
        if isinstance(check_result, CheckResult) and check_result.have_conditions():
            for cond in check_result.conditions_results:
                assert_that(
                    cond.category,
                    any_of(
                        ConditionCategory.PASS,
                        ConditionCategory.WARN,
                        ConditionCategory.FAIL
                    )
                )
