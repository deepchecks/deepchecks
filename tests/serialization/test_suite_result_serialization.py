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
"""CheckResult serialization tests."""
import json
import typing as t

from bs4 import BeautifulSoup
from hamcrest import (all_of, assert_that, calling, contains_exactly,
                      contains_string, equal_to, greater_than, has_entries,
                      has_length, has_property, instance_of, matches_regexp,
                      raises, starts_with, only_contains, has_item)
from ipywidgets import HTML, Tab, VBox
from wandb.sdk.data_types.base_types.wb_value import WBValue
from plotly.basedatatypes import BaseFigure
from IPython.display import Image

from deepchecks.core.check_result import CheckFailure, CheckResult
from deepchecks.core.serialization.common import (form_output_anchor,
                                                  plotlyjs_script)
from deepchecks.core.serialization.suite_result.html import \
    SuiteResultSerializer as HtmlSerializer
from deepchecks.core.serialization.suite_result.ipython import \
    SuiteResultSerializer as IPythonSerializer
from deepchecks.core.serialization.suite_result.json import \
    SuiteResultSerializer as JsonSerializer
from deepchecks.core.serialization.suite_result.wandb import \
    SuiteResultSerializer as WandbSerializer
from deepchecks.core.serialization.suite_result.widget import \
    SuiteResultSerializer as WidgetSerializer
from deepchecks.core.suite import SuiteResult
from deepchecks.utils.strings import get_random_string
from tests.serialization.test_check_failure_serialization import \
    assert_json_output as assert_check_failure_json_output
from tests.serialization.test_check_result_serialization import \
    assert_json_output as assert_check_result_json_output
from tests.serialization.utils import create_suite_result, instance_of_ipython_formatter


def test_html_serializer_initialization():
    serializer = HtmlSerializer(create_suite_result())


def test_html_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(HtmlSerializer).with_args(set()),
        raises(
            TypeError,
            'Expected "SuiteResult" but got "set"')
    )


def test_html_serialization():
    suite_result = create_suite_result()
    output = HtmlSerializer(suite_result).serialize()

    assert_that(
        output,
        all_of(
            instance_of(str),
            has_length(greater_than(0)),
            contains_string(f'<h1>{suite_result.name}</h1>'),
            contains_string('<h2>Conditions Summary</h2>'),
            contains_string('<h2>Check With Conditions Output</h2>'),
            contains_string('<h2>Check Without Conditions Output</h2>'),
            contains_string('<h2>Other Checks That Weren\'t Displayed</h2>'))
    )


def test_html_serialization_with__output_id__parameter():
    suite_result = create_suite_result()
    output_id = get_random_string(n=25)
    output = HtmlSerializer(suite_result).serialize(output_id=output_id)
    soup = BeautifulSoup(output, 'html.parser')

    assert_that(
        output,
        all_of(
            instance_of(str),
            has_length(greater_than(0))),
    )
    assert_that(
        are_navigation_links_present(soup, suite_result, output_id) is True
    )


def are_navigation_links_present(
    soup: BeautifulSoup,
    suite_result: SuiteResult,
    output_id: str,
) -> bool:
    summary_id = form_output_anchor(output_id)
    return all((
        soup.select_one(f'#{summary_id}') is not None,
        any(
            it.text == 'Go to top' and it.get('href') == f'#{summary_id}'
            for it in soup.select('a')
        ),
        all(
            soup.select_one(f'#{it.get_check_id(output_id)}') is not None
            for it in suite_result.results
            if isinstance(it, CheckResult)
        )
    ))


def test_html_serialization_with_plotply_activation_script():
    result = create_suite_result()
    output = HtmlSerializer(result).serialize()

    assert_that(
        output,
        all_of(
            instance_of(str),
            has_length(greater_than(0)),
            starts_with(plotlyjs_script()))
    )


def test_html_serialization_to_full_html_page():
    result = create_suite_result()
    output = HtmlSerializer(result).serialize(full_html=True)

    regexp = (
        r'^[\s]*<html>[\s]*'
        r'<head><meta charset="utf-8"\/><\/head>[\s]*'
        r'<body[\s]*(([\s\S\d\D\w\W]*))[\s]*>([\s\S\d\D\w\W]*)<\/body>[\s]*'
        r'<\/html>[\s]*$'
    )
    assert_that(
        output,
        all_of(instance_of(str), matches_regexp(regexp))
    )


# ============================================================================


def test_ipython_serializer_initialization():
    serializer = IPythonSerializer(create_suite_result())


def test_ipython_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(IPythonSerializer).with_args(set()),
        raises(
            TypeError,
            'Expected "SuiteResult" but got "set"')
    )


def test_ipython_serialization():
    suite_result = create_suite_result()
    output = IPythonSerializer(suite_result).serialize()

    assert_that(
        output,
        all_of(
            instance_of(list),
            has_length(greater_than(0)),
            only_contains(instance_of_ipython_formatter()),
            has_item(instance_of(Image)),
            has_item(instance_of(BaseFigure)))
    )

# ============================================================================


def test_json_serializer_initialization():
    serializer = JsonSerializer(create_suite_result())


def test_json_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(JsonSerializer).with_args(set()),
        raises(
            TypeError,
            'Expected "SuiteResult" but got "set"')
    )


def test_json_serialization():
    suite_result = create_suite_result()
    output = JsonSerializer(suite_result).serialize()

    # NOTE: SuiteResult JSON serializer returns not a json string
    # but a simple builtin python values which then can be serailized into json string.
    # We need to verify that this is actually true
    assert_that(
        json.loads(json.dumps(output)) == output
    )

    assert_that(
        output,
        all_of(
            instance_of(dict),
            has_entries({
                'name': instance_of(str),
                'results': all_of(
                    instance_of(list),
                    has_length(equal_to(len(suite_result.results))))
            }))
    )

    for index, payload in enumerate(output['results']):
        result = suite_result.results[index]
        if isinstance(result, CheckResult):
            assert_check_result_json_output(payload, suite_result.results[index])
        elif isinstance(result, CheckFailure):
            assert_check_failure_json_output(payload)
        else:
            raise TypeError(
                f'Suite contains results of unknown type - {type(result)}'
            )


# ============================================================================


def test_wandb_serializer_initialization():
    serializer = WandbSerializer(create_suite_result())


def test_wandb_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(WandbSerializer).with_args(set()),
        raises(
            TypeError,
            'Expected "SuiteResult" but got "set"')
    )


def test_wandb_serialization():
    suite_result = create_suite_result()
    output = WandbSerializer(suite_result).serialize()

    assert_that(output, instance_of(dict))

    for k, v in output.items():
        assert_that(k, starts_with(f'{suite_result.name}/'))
        assert_that(v, instance_of(WBValue))


# ============================================================================


def test_widget_serializer_initialization():
    serializer = WidgetSerializer(create_suite_result())


def test_widget_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(WidgetSerializer).with_args(set()),
        raises(
            TypeError,
            'Expected "SuiteResult" but got "set"')
    )


def test_widget_serialization():
    suite_result = create_suite_result()
    output = WidgetSerializer(suite_result).serialize()

    assert_that(
        output,
        all_of(
            instance_of(VBox),
            has_property(
                'children',
                all_of(
                    instance_of(tuple),
                    has_length(equal_to(3)),
                    contains_exactly(
                        instance_of(HTML),
                        instance_of(HTML),
                        instance_of(Tab)))))
    )
    assert_that(
        output.children[2], # tabs panel
        all_of(
            instance_of(Tab),
            has_property(
                'children',
                all_of(
                    instance_of(tuple),
                    has_length(equal_to(3)))))
    )
