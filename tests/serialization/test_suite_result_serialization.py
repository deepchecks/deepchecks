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
"""CheckResult serialization tests."""
import json
import typing as t

from bs4 import BeautifulSoup
from hamcrest import (all_of, assert_that, calling, contains_exactly, contains_string, equal_to, greater_than,
                      has_entries, has_item, has_length, has_property, instance_of, matches_regexp, only_contains,
                      raises, starts_with)
from IPython.display import Image
from ipywidgets import HTML, Accordion, Tab, VBox
from plotly.basedatatypes import BaseFigure
from wandb.sdk.data_types.base_types.wb_value import WBValue

from deepchecks.core.check_result import CheckFailure, CheckResult
from deepchecks.core.serialization.common import form_output_anchor, plotlyjs_script
from deepchecks.core.serialization.suite_result.html import SuiteResultSerializer as HtmlSerializer
from deepchecks.core.serialization.suite_result.ipython import SuiteResultSerializer as IPythonSerializer
from deepchecks.core.serialization.suite_result.json import SuiteResultSerializer as JsonSerializer
from deepchecks.core.serialization.suite_result.junit import SuiteResultSerializer as JunitSerializer
from deepchecks.core.serialization.suite_result.wandb import SuiteResultSerializer as WandbSerializer
from deepchecks.core.serialization.suite_result.widget import SuiteResultSerializer as WidgetSerializer
from deepchecks.core.suite import SuiteResult
from deepchecks.utils.strings import get_random_string
from tests.common import create_suite_result, instance_of_ipython_formatter
from tests.serialization.test_check_failure_serialization import assert_json_output as assert_check_failure_json_output
from tests.serialization.test_check_result_serialization import assert_json_output as assert_check_result_json_output


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
            if isinstance(it, CheckResult) and it.display and it.conditions_results
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
    suite_result = create_suite_result(
        include_results_without_conditions=False,
        include_results_without_display=False,
        include_results_without_conditions_and_display=False
    )
    output = JsonSerializer(suite_result).serialize()

    # NOTE: SuiteResult JSON serializer returns not a json string
    # but a simple builtin python values which then can be serailized into json string.
    # We need to verify that this is actually true
    assert_that(json.loads(json.dumps(output)) == output)

    assert_that(output, has_entries({
        'name': instance_of(str),
        'results': has_length(equal_to(len(suite_result.results)))
    }))

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


def test_junit_serializer_initialization():
    serializer = JunitSerializer(create_suite_result())


def test_junit_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(JunitSerializer).with_args(set()),
        raises(
            TypeError,
            'Expected "SuiteResult" but got "set"')
    )


def check_junit_test_suite(test_suite):
    assert_that(list(test_suite.attrib.keys()), ['errors', 'failures', 'name', 'tests', 'time', 'timestamp'])
    assert_that(test_suite.tag, 'testsuite')


def check_junit_test_case(test_case):
    assert_that(list(test_case.attrib.keys()), ['classname', 'name', 'time'])
    assert_that(test_case.tag, 'testcase')


def test_junit_serialization():
    suite_result = create_suite_result(
        include_results_without_conditions=False,
        include_results_without_display=False,
        include_results_without_conditions_and_display=False
    )
    output = JunitSerializer(suite_result).serialize()

    import xml.etree.ElementTree as ET

    formatted_response = ET.fromstring(output)

    assert_that(formatted_response.tag, 'testsuites')
    assert_that(list(formatted_response.attrib.keys()), ['errors', 'failures', 'name', 'tests', 'time'])

    for test_suite in list(formatted_response):
        check_junit_test_case(test_suite)

    for test_case in list(list(formatted_response)[0]):
        check_junit_test_case(test_case)


def test_junit_serialization_with_real_data(iris_split_dataset_and_model):
    from deepchecks.tabular.suites import full_suite

    ds_train, ds_test, rf_clf = iris_split_dataset_and_model

    suite = full_suite()

    results = suite.run(train_dataset=ds_train, test_dataset=ds_test, model=rf_clf)

    output = JunitSerializer(results).serialize()

    import xml.etree.ElementTree as ET

    formatted_response = ET.fromstring(output)

    assert_that(formatted_response.tag, 'testsuites')
    assert_that(list(formatted_response.attrib.keys()), ['errors', 'failures', 'name', 'tests', 'time'])

    for test_suite in list(formatted_response):
        check_junit_test_case(test_suite)

    for test_case in list(list(formatted_response)[0]):
        check_junit_test_case(test_case)


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

    top_level_accordion_assertion = all_of(
        instance_of(Accordion),
        has_property('children', contains_exactly(instance_of(VBox)))
    )
    section_assertion = all_of(
        instance_of(VBox),
        has_property('children', contains_exactly(
            instance_of(HTML),
            instance_of(Accordion)
        ))
    )
    content_assertion = all_of(
        instance_of(VBox),
        has_property('children', contains_exactly(
            instance_of(HTML),
            section_assertion,
            section_assertion,
            section_assertion,
            section_assertion,
        ))
    )

    assert_that(output, top_level_accordion_assertion)
    assert_that(output.children[0], content_assertion)
