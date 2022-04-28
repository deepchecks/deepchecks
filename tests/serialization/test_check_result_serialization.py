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
import typing as t
import json

import wandb
import pandas as pd
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure
from ipywidgets import VBox, HTML
from hamcrest import (
    assert_that,
    calling,
    raises,
    instance_of,
    all_of,
    contains_string,
    has_property,
    has_length,
    greater_than,
    matches_regexp,
    has_entries,
    equal_to,
    starts_with
)

from deepchecks.utils.strings import get_random_string
from deepchecks.core.serialization.common import plotlyjs_script
from deepchecks.core.serialization.check_result.json import display_from_json
from deepchecks.core.serialization.check_result.html import CheckResultSerializer as HtmlSerializer
from deepchecks.core.serialization.check_result.json import CheckResultSerializer as JsonSerializer
from deepchecks.core.serialization.check_result.wandb import CheckResultSerializer as WandbSerializer
from deepchecks.core.serialization.check_result.widget import CheckResultSerializer as WidgetSerializer

from tests.serialization.utils import DummyCheck
from tests.serialization.utils import create_check_result


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

    assert_that(
        output,
        all_of(
            instance_of(str),
            has_length(greater_than(0)),
            contains_string(result.get_header()),
            contains_string(t.cast(str, DummyCheck.__doc__)))
    )


def test_html_serialization_with_empty__check_sections__parameter():
    result = create_check_result()
    assert_that(
        calling(HtmlSerializer(result).serialize).with_args(check_sections=[]),
        raises(ValueError, 'include parameter cannot be empty')
    )


def test_html_serialization_with__output_id__parameter():
    result = create_check_result()
    output_id = get_random_string(n=25)
    output = HtmlSerializer(result).serialize(output_id=output_id)

    assert_that(
        output,
        all_of(
            instance_of(str),
            has_length(greater_than(0)),
            contains_string(output_id)),
    )


def test_check_result_without_display_and_conditions_into_html_serialization():
    check_result_without_conditions = create_check_result(include_conditions=False, include_display=False)
    output_without_conditions = HtmlSerializer(check_result_without_conditions).serialize()

    assert_that(
        output_without_conditions,
        all_of(instance_of(str), has_length(greater_than(0)))
    )

    full_check_result = create_check_result()
    full_output = HtmlSerializer(full_check_result).serialize()

    assert_that(
        full_output,
        all_of(instance_of(str), has_length(greater_than(0)))
    )

    assert_that(len(full_output) > len(output_without_conditions))



def test_html_serialization_with_plotply_activation_script():
    result = create_check_result()
    output = HtmlSerializer(result).serialize()

    assert_that(
        output,
        all_of(
            instance_of(str),
            has_length(greater_than(0)),
            starts_with(plotlyjs_script()))
    )


def test_html_serialization_to_full_html_page():
    result = create_check_result()
    output = HtmlSerializer(result).serialize(full_html=True)

    regexp = (
        r'^[\s]*<html>[\s]*'
        r'<head><meta charset="utf-8"\/><\/head>[\s]*'
        r'<body>([\s\S\d\D\w\W]*)<\/body>[\s]*'
        r'<\/html>[\s]*$'
    )

    assert_that(
        output,
        all_of(
            instance_of(str),
            matches_regexp(regexp)
        )
    )


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
            if isinstance(it, (pd.DataFrame, Styler)):
                assert_that(
                    output['display'][index],
                    all_of(
                        instance_of(dict),
                        has_entries({
                            'type': equal_to('dataframe'),
                            'payload': instance_of(list)
                        }))
                )
            elif isinstance(it, str):
                assert_that(
                    output['display'][index],
                    all_of(
                        instance_of(dict),
                        has_entries({
                            'type': equal_to('html'),
                            'payload': instance_of(str)
                        }))
                )
            elif isinstance(it, BaseFigure):
                assert_that(
                    output['display'][index],
                    all_of(
                        instance_of(dict),
                        has_entries({
                            'type': equal_to('plotly'),
                            'payload': instance_of(str)
                        }))
                )
            elif callable(it):
                assert_that(
                    output['display'][index],
                    all_of(
                        instance_of(dict),
                        has_entries({
                            'type': equal_to('images'),
                            'payload': instance_of(list)
                        }))
                )
            else:
                raise TypeError(f'Unknown display item type {type(it)}')


def test__display_from_json__function():
    output = JsonSerializer(create_check_result()).serialize()
    html = display_from_json(output)

    assert_that(
        html,
        all_of(
            instance_of(str),
            has_length(greater_than(0)),
            contains_string(output['check']['summary']),
            contains_string(output['header']))
    )


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
