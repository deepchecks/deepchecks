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

import pandas as pd
import wandb
from bs4 import BeautifulSoup
from hamcrest import *
from ipywidgets import HTML, VBox
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure

from deepchecks.core.check_result import DisplayMap
from deepchecks.core.serialization.check_result.html import CheckResultSerializer as HtmlSerializer
from deepchecks.core.serialization.check_result.json import CheckResultSerializer as JsonSerializer
from deepchecks.core.serialization.check_result.wandb import CheckResultSerializer as WandbSerializer
from deepchecks.core.serialization.check_result.widget import CheckResultSerializer as WidgetSerializer
from deepchecks.utils.strings import get_random_string
from tests.common import DummyCheck, create_check_result

# ===========================================


def test_html_serializer_initialization():
    serializer = HtmlSerializer(create_check_result())


def test_html_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(HtmlSerializer).with_args(5),
        raises(
            TypeError,
            'Expected "CheckResult" but got "int"')
    )


def test_html_serialization():
    result = create_check_result()
    output = HtmlSerializer(result).serialize()

    assert_that(output, instance_of(str))
    soup = BeautifulSoup(output, 'html.parser')

    assert_check_result_html_output_structure(soup)
    assert_style_loader(soup, 'presence')
    assert_plotly_loader(soup, 'presence')
    assert_chart_initializers(soup, 'presence')


def test_html_serialization_without_javascript():
    result = create_check_result()
    output = HtmlSerializer(result).serialize(use_javascript=False)

    assert_that(output, instance_of(str))

    soup = BeautifulSoup(output, 'html.parser')
    scripts = soup.select('script')
    assert_that(scripts, has_length(0))
    assert_check_result_html_output_structure(soup)


def test_html_serialization_with_empty__check_sections__parameter():
    result = create_check_result()
    assert_that(
        calling(HtmlSerializer(result).serialize).with_args(check_sections=[]),
        raises(ValueError, 'include parameter cannot be empty')
    )


def test_html_serialization_with__output_id__parameter():
    result = create_check_result()
    output_id = get_random_string(n=25)
    output = HtmlSerializer(result).serialize(output_id=output_id, embed_into_suite='suite')

    assert_that(output, instance_of(str))

    soup = BeautifulSoup(output, 'html.parser')
    assert_check_result_html_output_structure(soup)
    assert_style_loader(soup, 'absence')
    assert_plotly_loader(soup, 'absence')
    assert_chart_initializers(soup, 'presence')

    headers = [it for it in soup.select('h3[id]') if it.get('id', '').endswith(output_id)]
    links = [it for it in soup.select('a[href]') if it.get('href', '').endswith(output_id)]
    go_to_top_links = [it for it in links if it.text == 'Go to top']

    assert_that(headers, has_length(greater_than(0)))
    assert_that(links, has_length(greater_than(0)))
    assert_that(go_to_top_links, has_length(greater_than(0)))


def test_html_serialization_without_display_items_and_condition_results():
    check_result = create_check_result(include_conditions=False, include_display=False)
    output = HtmlSerializer(check_result).serialize()

    assert_that(output, instance_of(str))

    soup = BeautifulSoup(output, 'html.parser')
    assert_check_result_html_output_structure(
        soup,
        assert_additional_ouput='presence',  # section tag still will be present but it will be empty
        assert_condition_table='absence'
    )
    assert_style_loader(soup, 'presence')
    assert_plotly_loader(soup, 'absence')
    assert_chart_initializers(soup, 'absence')


def test_html_serialization_without_additional_output_section():
    check_result = create_check_result()
    output = HtmlSerializer(check_result).serialize(check_sections=['condition-table'])

    assert_that(output, instance_of(str))

    soup = BeautifulSoup(output, 'html.parser')
    assert_check_result_html_output_structure(
        soup,
        assert_additional_ouput='absence',
        assert_condition_table='presence'
    )
    assert_style_loader(soup, 'presence')
    assert_plotly_loader(soup, 'absence')
    assert_chart_initializers(soup, 'absence')


def test_html_serialization_without_condition_table_section():
    check_result = create_check_result()
    output = HtmlSerializer(check_result).serialize(check_sections=['additional-output'])

    assert_that(output, instance_of(str))

    soup = BeautifulSoup(output, 'html.parser')
    assert_check_result_html_output_structure(
        soup,
        assert_additional_ouput='presence',
        assert_condition_table='absence'
    )
    assert_style_loader(soup, 'presence')
    assert_plotly_loader(soup, 'presence')
    assert_chart_initializers(soup, 'presence')


def test_html_serialization_to_full_html_page():
    result = create_check_result()
    output = HtmlSerializer(result).serialize(full_html=True)

    assert_that(output, instance_of(str))

    soup = BeautifulSoup(output, 'html.parser')

    html = soup.select_one('html')
    assert_that(html, not_none())

    head = html.select_one('head')
    body = html.select_one('body')
    assert_that(head, not_none())
    assert_that(body, not_none())

    style = head.select('style')
    script = head.select('script')

    assert_that(style, not_none())
    assert_that(script, not_none())

    assert_check_result_html_output_structure(body)
    assert_style_loader(soup, what='absence')
    assert_plotly_loader(soup, what='absence')
    assert_chart_initializers(soup, what='presence')


def assert_check_result_html_output_structure(
    soup,
    assert_condition_table='presence',
    assert_additional_ouput='presence',
):
    if isinstance(soup, str):
        soup = BeautifulSoup(soup, 'html.parser')

    content = soup.select_one('article')
    assert_that(content, not_none())

    header = content.select_one('h3')
    assert_that(header, not_none())

    if assert_condition_table == 'presence':
        conditions_table = content.select_one('section[data-name="conditions-table"]')
        assert_that(conditions_table, not_none())
    elif assert_condition_table == 'absence':
        conditions_table = content.select_one('section[data-name="conditions-table"]')
        assert_that(conditions_table, none())
    else:
        raise RuntimeError(f'unknown value - {assert_condition_table}')

    if assert_additional_ouput == 'presence':
        additional_output = content.select_one('section[data-name="additional-output"]')
        assert_that(additional_output, not_none())
    elif assert_additional_ouput == 'absence':
        additional_output = content.select_one('section[data-name="additional-output"]')
        assert_that(additional_output, none())
    else:
        raise RuntimeError(f'unknown value - {assert_condition_table}')


def assert_style_loader(soup, what='presence'):
    if isinstance(soup, str):
        soup = BeautifulSoup(soup, 'html.parser')

    if what == 'presence':
        tags = soup.select('script#deepchecks-style-loader')
        assert_that(tags, has_length(1))
    elif what == 'absence':
        tags = soup.select('script#deepchecks-style-loader')
        assert_that(tags, has_length(0))
    else:
        raise RuntimeError(f'unknown value - {what}')


def assert_plotly_loader(soup, what='presence'):
    if isinstance(soup, str):
        soup = BeautifulSoup(soup, 'html.parser')

    if what == 'presence':
        tags = soup.select('script#deepchecks-plotly-src')
        assert_that(tags, has_length(1))
        tags = soup.select('script#deepchecks-plotly-loader')
        assert_that(tags, has_length(1))
    elif what == 'absence':
        tags = soup.select('script#deepchecks-plotly-src')
        assert_that(tags, has_length(0))
        tags = soup.select('script#deepchecks-plotly-loader')
        assert_that(tags, has_length(0))
    else:
        raise RuntimeError(f'unknown value - {what}')


def assert_chart_initializers(soup, what='presence'):
    if isinstance(soup, str):
        soup = BeautifulSoup(soup, 'html.parser')

    if what == 'presence':
        tags = soup.select('script#deepchecks-plot-initializer')
        assert_that(tags, has_length(greater_than(0)))
    elif what == 'absence':
        tags = soup.select('script#deepchecks-plot-initializer')
        assert_that(tags, has_length(0))
    else:
        raise RuntimeError(f'unknown value - {what}')


# ===========================================


def test_json_serializer_initialization():
    serializer = JsonSerializer(create_check_result())


def test_json_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(JsonSerializer).with_args(dict()),
        raises(
            TypeError,
            'Expected "CheckResult" but got "dict"')
    )


def test_json_serialization():
    check_result = create_check_result()
    output = JsonSerializer(check_result).serialize()
    assert_json_output(output, check_result)


def test_check_result_without_conditions_and_display_into_json_serialization():
    check_result = create_check_result(include_conditions=False, include_display=False)
    output = JsonSerializer(check_result).serialize()
    assert_json_output(
        output,
        check_result,
        with_conditions_table=False,
        with_display=False
    )


def assert_json_output(
    output,
    check_result,
    with_conditions_table=True,
    with_display=True
):
    # NOTE: CheckResult JSON serializer returns not a json string
    # but a simple builtin python values which then can be serailized into json string.
    # We need to verify that this is actually true
    assert_that(
        json.loads(json.dumps(output)) == output
    )

    assert_that(
        output, all_of(
            instance_of(dict),
            has_entries({
                'check': instance_of(dict),
                'value': equal_to(check_result.value),
                'header': equal_to(check_result.header),
                'conditions_results': all_of(
                    instance_of(list),
                    has_length(greater_than(0))
                    if with_conditions_table is True
                    else has_length(equal_to(0))
                ),
                'display': all_of(
                    instance_of(list),
                    has_length(greater_than(0))
                    if with_display is True
                    else has_length(equal_to(0))
                )
            }))
    )

    if with_display is True:
        for index, it in enumerate(check_result.display):
            assert_that(
                output['display'][index],
                is_serialized_to_json_display_item(type(it))
            )


def is_serialized_to_json_display_item(item_type):
    if issubclass(item_type, (pd.DataFrame, Styler)):
        return has_entries({
            'type': equal_to('dataframe'),
            'payload': instance_of(list)
        })
    elif issubclass(item_type, str):
        return has_entries({
            'type': equal_to('html'),
            'payload': instance_of(str)
        })
    elif issubclass(item_type, BaseFigure):
        return has_entries({
            'type': equal_to('plotly'),
            'payload': instance_of(str)
        })
    elif issubclass(item_type, t.Callable):
        return has_entries({
            'type': equal_to('images'),
            'payload': all_of(has_length(greater_than(0))),
        })
    elif issubclass(item_type, DisplayMap):
        return has_entries({
            'type': equal_to('displaymap'),
            'payload': instance_of(dict),
        })
    else:
        raise TypeError(f'Unknown display item type {type(item_type)}')


# ===========================================


def test_wandb_serializer_initialization():
    serializer = WandbSerializer(create_check_result())


def test_wandb_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(WandbSerializer).with_args(dict()),
        raises(
            TypeError,
            'Expected "CheckResult" but got "dict"')
    )


def test_wandb_serialization():
    check_result = create_check_result()
    output = WandbSerializer(check_result).serialize()

    assert_that(
        output,
        wandb_output_assertion(check_result)
    )


def test_check_result_without_conditions_serialization_to_wandb():
    check_result = create_check_result(include_conditions=False)
    output = WandbSerializer(check_result).serialize()

    assert_that(
        output,
        wandb_output_assertion(check_result, with_conditions_table=False)
    )


def test_check_result_without_conditions_and_display_serialization_to_wandb():
    check_result = create_check_result(include_conditions=False, include_display=False)
    output = WandbSerializer(check_result).serialize()

    assert_that(
        output,
        wandb_output_assertion(
            check_result,
            with_conditions_table=False,
            with_display=False)
    )


def wandb_output_assertion(
    check_result,
    with_conditions_table=True,
    with_display=True
):
    entries = {
        f'{check_result.header}/results': instance_of(wandb.Table),
    }

    if with_display is True:
        for index, it in enumerate(check_result.display):
            if isinstance(it, (pd.DataFrame, Styler)):
                entries[f'{check_result.header}/item-{index}-table'] = instance_of(wandb.Table)
            elif isinstance(it, str):
                entries[f'{check_result.header}/item-{index}-html'] = instance_of(wandb.Html)
            elif isinstance(it, BaseFigure):
                entries[f'{check_result.header}/item-{index}-plot'] = instance_of(wandb.Plotly)
            elif callable(it):
                entries[f'{check_result.header}/item-{index}-figure'] = instance_of(wandb.Image)
            elif isinstance(it, DisplayMap):
                # TODO:
                pass
            else:
                raise TypeError(f'Unknown display item type {type(it)}')

    if with_conditions_table is True:
        entries[f'{check_result.header}/conditions table'] = instance_of(wandb.Table)

    return all_of(instance_of(dict), has_entries(entries))


# ===========================================


def test_widget_serializer_initialization():
    serializer = WandbSerializer(create_check_result())


def test_widget_serializer_initialization_with_incorrect_type_of_value():
    assert_that(
        calling(WandbSerializer).with_args(dict()),
        raises(
            TypeError,
            'Expected "CheckResult" but got "dict"')
    )


def test_widget_serialization():
    check_result = create_check_result()
    output = WidgetSerializer(check_result).serialize()
    assert_widget_output(output, check_result)


def test_widget_serialization_without_conditions_section():
    check_result = create_check_result()
    output = WidgetSerializer(check_result).serialize(check_sections=['additional-output'])
    assert_widget_output(
        output,
        check_result,
        with_conditions_section=False,
        with_display_section=True
    )


def test_widget_serialization_without_display_section():
    check_result = create_check_result()
    output = WidgetSerializer(check_result).serialize(check_sections=['condition-table'])
    assert_widget_output(
        output,
        check_result,
        with_conditions_section=True,
        with_display_section=False
    )


def test_widget_serialization_with_empty__check_sections__parameter():
    result = create_check_result()
    assert_that(
        calling(WidgetSerializer(result).serialize).with_args(check_sections=[]),
        raises(ValueError, 'include parameter cannot be empty')
    )


def assert_widget_output(
    output,
    check_result,
    with_conditions_section=True,
    with_display_section=True
):
    children_count = 4
    header_section_index = 0
    summary_section_index = 1
    conditions_section_index = 2
    display_section_index = 3

    if with_conditions_section is False and with_display_section is False:
        children_count = 2
    elif with_conditions_section is False and with_display_section is True:
        children_count = 3
        display_section_index = 2
    elif with_conditions_section is False or with_display_section is False:
        children_count = 3

    assert_that(
        output,
        all_of(
            instance_of(VBox),
            has_property(
                'children',
                all_of(
                    instance_of(tuple),
                    has_length(children_count))))
    )

    assert_that(
        output.children[header_section_index],
        all_of(
            instance_of(HTML),
            has_property(
                'value',
                all_of(
                    instance_of(str),
                    contains_string(check_result.header))))
    )
    assert_that(
        output.children[summary_section_index],
        all_of(
            instance_of(HTML),
            has_property(
                'value',
                all_of(
                    instance_of(str),
                    contains_string(t.cast(str, DummyCheck.__doc__)))))
    )

    if with_conditions_section is True:
        assert_that(
            output.children[conditions_section_index],
            instance_of(HTML)
        )

    if with_display_section is True:
        assert_that(
            output.children[display_section_index],
            all_of(
                instance_of(VBox),
                has_property(
                    'children',
                    all_of(
                        instance_of(tuple),
                        has_length(equal_to(len(check_result.display) + 1)))))  # plus header element
        )
