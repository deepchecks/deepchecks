# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Common functions."""
import typing as t

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import this
from hamcrest import any_of, instance_of

from deepchecks.core.check_result import CheckFailure, CheckResult
from deepchecks.core.checks import BaseCheck
from deepchecks.core.condition import ConditionCategory, ConditionResult
from deepchecks.core.serialization.abc import (HTMLFormatter,
                                               IPythonDisplayFormatter,
                                               JPEGFormatter, JSONFormatter,
                                               MarkdownFormatter,
                                               MimeBundleFormatter,
                                               PNGFormatter)
from deepchecks.core.suite import SuiteResult


class DummyCheck(BaseCheck):
    """Dummy check type for testing purpose."""

    def run(self, *args, **kwargs):
        raise NotImplementedError()


def create_suite_result(
    name: str = 'Dummy Suite Result',
    n_of_results: int = 5,
    n_of_failures: int = 5
) -> SuiteResult:
    results = [
        create_check_result(value=i, header=f'Dummy Result #{i}')
        for i in range(n_of_results)
    ]
    failures = [
        CheckFailure(DummyCheck(), Exception(f'Error #{i}'))
        for i in range(n_of_failures)
    ]
    return SuiteResult(
        name=name,
        results=[*results, *failures],
        extra_info=this.s.splitlines()
    )


def create_check_result(
    value: t.Any = None,
    header: str = 'Dummy Result',
    include_display: bool = True,
    include_conditions: bool = True
) -> CheckResult:
    def draw_plot():
        plt.subplots()
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])

    plotly_figure = px.bar(
        px.data.gapminder().query("country == 'Canada'"),
        x='year', y='pop'
    )
    display = [
        header,
        pd.DataFrame({'foo': range(10), 'bar': range(10)}),
        pd.DataFrame({'foo': range(10), 'bar': range(10)}).style,
        plotly_figure,
        draw_plot
    ]
    result = CheckResult(
        value=value or 1000,
        header='Dummy Result',
        display=display if include_display else None,
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
