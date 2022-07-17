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
from hamcrest import *
from IPython.display import Image
from ipywidgets import HTML, Accordion, Tab, VBox
from plotly.basedatatypes import BaseFigure
from wandb.sdk.data_types.base_types.wb_value import WBValue

from deepchecks.core.check_result import CheckFailure, CheckResult
from deepchecks.core.serialization.common import form_output_anchor
from deepchecks.core.serialization.suite_result.html import SuiteResultSerializer as HtmlSerializer
from deepchecks.core.serialization.suite_result.json import SuiteResultSerializer as JsonSerializer
from deepchecks.core.serialization.suite_result.wandb import SuiteResultSerializer as WandbSerializer
from deepchecks.core.serialization.suite_result.widget import SuiteResultSerializer as WidgetSerializer
from deepchecks.core.suite import SuiteResult
from deepchecks.utils.strings import get_random_string
from tests.common import create_suite_result, instance_of_ipython_formatter
from tests.serialization.test_check_failure_serialization import assert_json_output as assert_check_failure_json_output
from tests.serialization.test_check_result_serialization import assert_chart_initializers
from tests.serialization.test_check_result_serialization import assert_json_output as assert_check_result_json_output
from tests.serialization.test_check_result_serialization import assert_plotly_loader, assert_style_loader


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

    assert_that(output, instance_of(str))

    soup = BeautifulSoup(output, 'html.parser')
    assert_suite_result_output_structure(soup)
    assert_plotly_loader(soup, 'presence')
    assert_style_loader(soup, 'presence')


def test_html_serialization_with__output_id__parameter():
    output_id = get_random_string(n=25)
    suite_result = create_suite_result()
    output = HtmlSerializer(suite_result).serialize(output_id=output_id)

    assert_that(output, instance_of(str))

    soup = BeautifulSoup(output, 'html.parser')
    assert_suite_result_output_structure(soup)
    assert_plotly_loader(soup, 'presence')
    assert_style_loader(soup, 'presence')

    links = soup.select('a')
    navigation_links = [l for l in links if l.get('href', '').endswith(output_id)]
    gototop_links = [l for l in navigation_links if l.text == 'Go to top']

    assert_that(navigation_links, has_length(greater_than(0)))
    assert_that(gototop_links, has_length(greater_than(0)))


def test_html_serialization_without__output_id__parameter():
    suite_result = create_suite_result()
    output = HtmlSerializer(suite_result).serialize()

    assert_that(output, instance_of(str))

    soup = BeautifulSoup(output, 'html.parser')
    assert_suite_result_output_structure(soup)
    assert_plotly_loader(soup, 'presence')
    assert_style_loader(soup, 'presence')

    links = soup.select('a')
    navigation_links = [l for l in links if l.get('href', '').startswith('#')]
    gototop_links = [l for l in navigation_links if l.text == 'Go to top']

    assert_that(navigation_links, has_length(equal_to(0)))
    assert_that(gototop_links, has_length(equal_to(0)))


def test_html_serialization_to_full_html_page():
    suite_result = create_suite_result()
    output = HtmlSerializer(suite_result).serialize(full_html=True)

    assert_that(output, instance_of(str))
    soup = BeautifulSoup(output, 'html.parser')

    html = soup.select_one('html')
    assert_that(html, not_none())

    head = html.select_one('head')
    assert_that(head, not_none())

    script = head.select_one('script')
    style = head.select_one('style')
    assert_that(script, not_none())
    assert_that(style, not_none())

    body = html.select_one('body')
    assert_that(body, not_none())


def test_html_serialization_with_output_embeded_into_iframe():
    suite_result = create_suite_result()
    output = HtmlSerializer(suite_result).serialize(embed_into_iframe=True)

    assert_that(output, instance_of(str))
    soup = BeautifulSoup(output, 'html.parser')

    iframe = soup.select_one('iframe')
    assert_that(iframe, not_none())
    assert_that(iframe.get('srcdoc'), not_none())


def test_html_serialization_with__use_javascript__set_to_false():
    suite_result = create_suite_result()
    output = HtmlSerializer(suite_result).serialize(use_javascript=False)

    assert_that(output, instance_of(str))
    soup = BeautifulSoup(output, 'html.parser')
    assert_suite_result_output_structure(soup)

    scripts = soup.select('script')
    assert_that(scripts, has_length(equal_to(0)))


def assert_suite_result_output_structure(soup):
    if isinstance(soup, str):
        soup = BeautifulSoup(soup, 'html.parser')

    suite_accordion = soup.select_one('details[data-name="suite-result"].deepchecks-collapsible')
    assert_that(suite_accordion, not_none())

    suite_accordion_title = suite_accordion.select_one('summary')
    suite_accordion_content = suite_accordion.select_one('div.deepchecks-collapsible-content')

    assert_that(suite_accordion_title, not_none())
    assert_that(suite_accordion_content, not_none())

    header_tag = suite_accordion_content.select_one('h1:first-child')
    assert_that(header_tag, not_none())

    detail_tags = suite_accordion_content.select('details[data-name="check-results-list"].deepchecks-collapsible')
    assert_that(detail_tags, has_length(greater_than_or_equal_to(4)))

    f, s, t, fr, *_ = detail_tags

    assert_that(f.select_one('summary'), not_none())
    assert_that(f.select_one('summary').text, equal_to('Didn`t Pass'))
    assert_that(f.select_one('div.deepchecks-collapsible-content'), not_none())

    assert_that(s.select_one('summary'), not_none())
    assert_that(s.select_one('summary').text, equal_to('Passed'))
    assert_that(s.select_one('div.deepchecks-collapsible-content'), not_none())

    assert_that(t.select_one('summary'), not_none())
    assert_that(t.select_one('summary').text, equal_to('Other'))
    assert_that(t.select_one('div.deepchecks-collapsible-content'), not_none())

    assert_that(fr.select_one('summary'), not_none())
    assert_that(fr.select_one('summary').text, equal_to('Didn`t Run'))
    assert_that(fr.select_one('div.deepchecks-collapsible-content'), not_none())


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
