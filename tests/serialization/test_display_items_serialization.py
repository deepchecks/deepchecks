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
"""Display items serialization tests."""
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import wandb
from bs4 import BeautifulSoup
from hamcrest import *
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure

from deepchecks.core.check_result import DisplayMap
from deepchecks.core.serialization.check_result.html import DisplayItemsSerializer as HtmlSerializer
from deepchecks.core.serialization.check_result.json import DisplayItemsSerializer as JsonSerializer
from deepchecks.core.serialization.check_result.wandb import DisplayItemsSerializer as WandbSerializer
from tests.common import create_check_result_display

# =========================================================================

def test_string_serialization_to_html():
    value = '<h1>Hello world</h1>'
    output = HtmlSerializer([value]).serialize()
    assert_that(output, contains_exactly(equal_to(f'<div>{value}</div>')))


def test_dataframe_serialization_to_html():
    value = pd.DataFrame({'foo': [1, 2, 3], 'bar': [1,2,3]})
    output = HtmlSerializer([value]).serialize()

    assert_that(output, contains_exactly(instance_of(str)))

    soup = BeautifulSoup(output[0], 'html.parser')
    columns = [it.text for it in soup.select('table > thead > tr > th')]
    rows = soup.select('table > thead > tr > th')

    assert_that(columns, contains_exactly(instance_of(str), equal_to('foo'), equal_to('bar')))
    assert_that(rows, has_length(3))


def test_plotly_serialization_to_html():
    plot = px.bar(px.data.gapminder().query("country == 'Canada'"), x='year', y='pop')
    output = HtmlSerializer([plot]).serialize()

    assert_that(output, contains_exactly(instance_of(str)))

    soup = BeautifulSoup(output[0], 'html.parser')
    plot_container = soup.select_one('div[id]')

    assert_that(plot_container, not_none())
    assert_that(plot_container.get('id'), starts_with('deepchecks-'))

    script = soup.select_one('script')
    assert_that(script, not_none())


def test_plotly_serialization_to_html_image_tag():
    plot = px.bar(px.data.gapminder().query("country == 'Canada'"), x='year', y='pop')
    output = HtmlSerializer([plot]).serialize(use_javascript=False)

    assert_that(output, contains_exactly(instance_of(str)))

    soup = BeautifulSoup(output[0], 'html.parser')
    imgtag = soup.select_one('img')

    assert_that(imgtag, all_of(
        not_none(),
        has_property('attrs', has_entry('src', starts_with('data:image/png;base64')))
    ))


def test_matplot_serialization_to_html():
    def value():
        plt.subplots()
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.subplots()
        plt.plot([1, 2, 3, 4], [1, 2, 6, 1])

    output = HtmlSerializer([value]).serialize(use_javascript=False)

    assert_that(output, contains_exactly(instance_of(str)))
    soup = BeautifulSoup(output[0], 'html.parser')

    assert_that(soup.select('img'), contains_exactly(
        has_property('attrs', has_entry('src', starts_with('data:image/png;base64'))),
        has_property('attrs', has_entry('src', starts_with('data:image/png;base64')))
    ))


def test_displaymap_serialization_to_html():
    value = DisplayMap(foo=create_check_result_display(), bar=create_check_result_display())
    output = HtmlSerializer([value]).serialize()

    assert_that(output, contains_exactly(instance_of(str)))
    soup = BeautifulSoup(output[0], 'html.parser')

    tabs_panel = soup.select_one('div.deepchecks-tabs')
    assert_that(tabs_panel, not_none())

    tab_btns = [it.text for it in tabs_panel.select('div.deepchecks-tabs-btns > button')]
    tab_contents = tabs_panel.select('div.deepchecks-tab')

    assert_that(tab_btns, contains_exactly(*[equal_to(k) for k in value.keys()]))
    assert_that(tab_contents, has_length(equal_to(len(value))))


def test_displaymap_serialization_to_html_without_javascript():
    value = DisplayMap(foo=create_check_result_display(), bar=create_check_result_display())
    output = HtmlSerializer([value]).serialize(use_javascript=False)

    assert_that(output, contains_exactly(instance_of(str)))
    soup = BeautifulSoup(output[0], 'html.parser')

    details_tags = soup.select('details')
    summary_tags = [it.select_one('summary').text for it in details_tags]

    assert_that(details_tags, has_length(len(value)))
    assert_that(summary_tags, contains_exactly(*[equal_to(k) for k in value.keys()]))


def test_unknown_value_serialization_to_html():
    assert_that(
        calling(HtmlSerializer([object()]).serialize),
        raises(TypeError, 'Unable to handle display item of type: <class \'object\'>')
    )


# =========================================================================


def test_string_serialization_to_json():
    value = '<h1>Hello world</h1>'
    output = JsonSerializer([value]).serialize()
    assert_that(output, contains_exactly(has_entries({'type': 'html','payload': equal_to(value)})))


def test_dataframe_serialization_to_json():
    value = pd.DataFrame({'foo': [1, 2, 3], 'bar': [1,2,3]})
    output = JsonSerializer([value]).serialize()

    assert_that(output, contains_exactly(has_entries({
        'type': 'dataframe',
        'payload': all_of(instance_of(list), has_length(3))
    })))


def test_plotly_serialization_to_json():
    plot = px.bar(px.data.gapminder().query("country == 'Canada'"), x='year', y='pop')
    output = JsonSerializer([plot]).serialize()

    assert_that(output, contains_exactly(has_entries({
        'type': 'plotly',
        'payload': instance_of(str)
    })))


def test_matplot_serialization_to_json():
    def value():
        plt.subplots()
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.subplots()
        plt.plot([1, 2, 3, 4], [1, 2, 6, 1])

    output = JsonSerializer([value]).serialize(use_javascript=False)

    assert_that(output, contains_exactly(has_entries({
        'type': 'images',
        'payload': all_of(instance_of(list), has_length(2))
    })))


def test_displaymap_serialization_to_json():
    value = DisplayMap(foo=create_check_result_display(), bar=create_check_result_display())
    output = JsonSerializer([value]).serialize()

    assert_that(output, contains_exactly(has_entries({
        'type': 'displaymap',
        'payload': instance_of(dict)
    })))

    payload = output[0]['payload']

    for name, display_items in value.items():
        items_assertion = []

        for it in display_items:
            if isinstance(it, str):
                items_assertion.append(has_entries({
                    'type': 'html',
                    'payload': instance_of(str)
                }))
            elif isinstance(it, (pd.DataFrame, Styler)):
                items_assertion.append(has_entries({
                    'type': 'dataframe',
                    'payload': instance_of(list)
                }))
            elif isinstance(it, BaseFigure):
                items_assertion.append(has_entries({
                    'type': 'plotly',
                    'payload': instance_of(str)
                }))
            elif isinstance(it, DisplayMap):
                items_assertion.append(has_entries({
                    'type': 'displaymap',
                    'payload': instance_of(dict)
                }))
            elif callable(it):
                items_assertion.append(has_entries({
                    'type': 'images',
                    'payload': instance_of(list)
                }))
            else:
                raise RuntimeError(f'Unknown display item type - {type(it)}')

        assert_that(payload, has_entry(name, contains_exactly(*items_assertion)))


def test_unknown_value_serialization_to_json():
    assert_that(
        calling(JsonSerializer([object()]).serialize),
        raises(TypeError, 'Unable to handle display item of type: <class \'object\'>')
    )


# =========================================================================


def test_string_serialization_to_wandb():
    value = '<h1>Hello world</h1>'
    output = WandbSerializer([value]).serialize()

    assert_that(output, contains_exactly(all_of(
        instance_of(tuple),
        contains_exactly(
            equal_to('item-0-html'),
            all_of(
                instance_of(wandb.Html),
                has_property('html', contains_string(value))
            )
        )
    )))


def test_dataframe_serialization_to_wandb():
    value = pd.DataFrame({'foo': [1, 2, 3], 'bar': [1,2,3]})
    output = WandbSerializer([value]).serialize()

    assert_that(output, contains_exactly(all_of(
        instance_of(tuple),
        contains_exactly(equal_to('item-0-table'), instance_of(wandb.Table))
    )))


def test_plotly_serialization_to_wandb():
    plot = px.bar(px.data.gapminder().query("country == 'Canada'"), x='year', y='pop')
    output = WandbSerializer([plot]).serialize()

    assert_that(output, contains_exactly(all_of(
        instance_of(tuple),
        contains_exactly(equal_to('item-0-plot'), instance_of(wandb.Plotly))
    )))


def test_matplot_serialization_to_wandb():
    def value():
        plt.subplots()
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.subplots()
        plt.plot([1, 2, 3, 4], [1, 2, 6, 1])

    output = WandbSerializer([value]).serialize(use_javascript=False)

    assert_that(output, contains_exactly(all_of(
        instance_of(tuple),
        contains_exactly(equal_to('item-0-figure'), instance_of(wandb.Image))
    )))


def test_displaymap_serialization_to_wandb():
    value = DisplayMap(foo=create_check_result_display(), bar=create_check_result_display())
    output = WandbSerializer([value]).serialize()

    assertions = []

    for name, display_items in value.items():
        for index, item in enumerate(display_items):
            if isinstance(item, str):
                assertions.append(all_of(
                    instance_of(tuple),
                    contains_exactly(
                        f'item-0-displaymap/{name}/item-{index}-html',
                        instance_of(wandb.Html)
                    )
                ))
            elif isinstance(item, (pd.DataFrame, Styler)):
                assertions.append(all_of(
                    instance_of(tuple),
                    contains_exactly(
                        f'item-0-displaymap/{name}/item-{index}-table',
                        instance_of(wandb.Table)
                    )
                ))
            elif isinstance(item, BaseFigure):
                assertions.append(all_of(
                    instance_of(tuple),
                    contains_exactly(
                        f'item-0-displaymap/{name}/item-{index}-plot',
                        instance_of(wandb.Plotly)
                    )
                ))
            elif isinstance(item, DisplayMap):
                pass # TODO: value does not contain nested displaymap
            elif callable(item):
                assertions.append(all_of(
                    instance_of(tuple),
                    contains_exactly(
                        f'item-0-displaymap/{name}/item-{index}-figure',
                        instance_of(wandb.Image)
                    )
                ))

    assert_that(output, contains_exactly(*assertions))


def test_unknown_value_serialization_to_wandb():
    assert_that(
        calling(WandbSerializer([object()]).serialize),
        raises(TypeError, 'Unable to handle display item of type: <class \'object\'>')
    )
