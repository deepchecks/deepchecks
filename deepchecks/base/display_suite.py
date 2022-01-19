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
"""Handle display of suite result."""
from typing import List, Union
import itertools
import os
import sys
import re

# pylint: disable=protected-access
import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import pandas as pd
from IPython.core.display import display, display_html
from IPython import get_ipython
import ipywidgets as widgets
from ipywidgets.embed import embed_minimal_html

from deepchecks import errors
from deepchecks.utils.ipython import is_widgets_enabled
from deepchecks.utils.strings import get_random_string
from deepchecks.base.check import CheckResult, CheckFailure
from deepchecks.base.display_pandas import dataframe_to_html, get_conditions_table, \
                                           get_result_navigation_display


__all__ = ['display_suite_result', 'ProgressBar']


_CONDITIONS_SUMMARY_TITLE = '<h2>Conditions Summary</h2>'
_NO_CONDITIONS_SUMMARY_TITLE = '<p>No conditions defined on checks in the suite.</p>'
_NO_OUTPUT_TEXT = '<p>No outputs to show.</p>'
_CHECKS_WITH_CONDITIONS_TITLE = '<h2>Check With Conditions Output</h2>'
_CHECKS_WITHOUT_CONDITIONS_TITLE = '<h2>Check Without Conditions Output</h2>'
_CHECKS_WITHOUT_DISPLAY_TITLE = '<h2>Other Checks That Weren\'t Displayed</h2>'


def _get_check_widget(check_res: CheckResult, unique_id: str) -> widgets.VBox:
    return check_res.display_check(unique_id=unique_id, as_widget=True)


def _add_widget_classes(widget: widgets.HTML):
    """Add classes of regular jupyter output (makes dataframe and links look better)."""
    widget.add_class('rendered_html')
    widget.add_class('jp-RenderedHTMLCommon')
    widget.add_class('jp-RenderedHTML')
    widget.add_class('jp-OutputArea-output')


def _create_table_widget(df_html: str) -> widgets.VBox:
    table_box = widgets.VBox()
    df_widg = widgets.HTML(df_html)
    table_box.children = [df_widg]
    _add_widget_classes(table_box)
    return table_box


class ProgressBar:
    """Progress bar for display while running suite."""

    def __init__(self, name, length):
        """Initialize progress bar."""
        shared_args = {'total': length, 'desc': name, 'unit': ' Check', 'leave': False, 'file': sys.stdout}
        if is_widgets_enabled():
            self.pbar = tqdm_notebook(**shared_args, colour='#9d60fb')
        else:
            # Normal tqdm with colour in notebooks produce bug that the cleanup doesn't remove all characters. so
            # until bug fixed, doesn't add the colour to regular tqdm
            self.pbar = tqdm.tqdm(**shared_args, bar_format=f'{{l_bar}}{{bar:{length}}}{{r_bar}}')

    def set_text(self, text):
        """Set current running check."""
        self.pbar.set_postfix(Check=text)

    def close(self):
        """Close the progress bar."""
        self.pbar.close()

    def inc_progress(self):
        """Increase progress bar value by 1."""
        self.pbar.update(1)


def get_display_exists_icon(exists: bool):
    if exists:
        return '<div style="text-align: center">Yes</div>'
    return '<div style="text-align: center">No</div>'


def _display_suite_widgets(summary: str,
                           unique_id: str,
                           checks_with_conditions: List[CheckResult],
                           checks_wo_conditions_display: List[CheckResult],
                           checks_w_condition_display: List[CheckResult],
                           others_table: List,
                           light_hr: str,
                           html_out):  # pragma: no cover
    """Display results of suite in as Tab widget."""
    tab = widgets.Tab()
    condition_tab = widgets.VBox()
    _add_widget_classes(condition_tab)
    checks_wo_tab = widgets.VBox()
    _add_widget_classes(checks_wo_tab)
    others_tab = widgets.VBox()
    tab.children = [condition_tab, checks_wo_tab, others_tab]
    tab.set_title(0, 'Checks With Conditions')
    tab.set_title(1, 'Checks Without Conditions')
    tab.set_title(2, 'Checks Without Output')

    if checks_with_conditions:
        cond_html_table = dataframe_to_html(get_conditions_table(checks_with_conditions, unique_id, 300))
        h2_widget = widgets.HTML(_CONDITIONS_SUMMARY_TITLE)
        condition_tab_children = [h2_widget, _create_table_widget(cond_html_table)]
    else:
        condition_tab_children = [widgets.HTML(_NO_CONDITIONS_SUMMARY_TITLE)]

    condition_tab_children.append(widgets.HTML(_CHECKS_WITH_CONDITIONS_TITLE))
    if checks_w_condition_display:
        for i, r in enumerate(checks_w_condition_display):
            condition_tab_children.append(_get_check_widget(r, unique_id))
            if i < len(checks_w_condition_display) - 1:
                condition_tab_children.append(widgets.HTML(light_hr))
    else:
        condition_tab_children.append(widgets.HTML(_NO_OUTPUT_TEXT))

    checks_wo_tab_children = []
    checks_wo_tab_children.append(widgets.HTML(_CHECKS_WITHOUT_CONDITIONS_TITLE))
    if checks_wo_conditions_display:
        if unique_id:
            nav_table = get_result_navigation_display(checks_wo_conditions_display, unique_id)
            checks_wo_tab_children.append(widgets.HTML(nav_table))
            checks_wo_tab_children.append(widgets.HTML(light_hr))
        for i, r in enumerate(checks_wo_conditions_display):
            checks_wo_tab_children.append(_get_check_widget(r, unique_id))
            if i < len(checks_wo_conditions_display) - 1:
                checks_wo_tab_children.append(widgets.HTML(light_hr))
    else:
        checks_wo_tab_children.append(widgets.HTML(_NO_OUTPUT_TEXT))

    if others_table:
        others_table = pd.DataFrame(data=others_table, columns=['Check', 'Reason', 'sort'])
        others_table.sort_values(by=['sort'], inplace=True)
        others_table.drop('sort', axis=1, inplace=True)
        others_df = dataframe_to_html(others_table.style.hide_index())
        h2_widget = widgets.HTML(_CHECKS_WITHOUT_DISPLAY_TITLE)
        others_tab.children = [h2_widget, _create_table_widget(others_df)]
    else:
        others_tab.children = [widgets.HTML(_NO_OUTPUT_TEXT)]
    condition_tab.children = condition_tab_children
    checks_wo_tab.children = checks_wo_tab_children

    tab_css = '<style>.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {flex: 0 1 auto}</style>'
    page = widgets.VBox()
    page.children = [widgets.HTML(summary), widgets.HTML(tab_css), tab]
    if html_out:
        if isinstance(html_out, str):
            if '.' in html_out:
                basename, ext = html_out.rsplit('.', 1)
            else:
                basename = html_out
                ext = 'html'
            html_out = f'{basename}.{ext}'
            c = itertools.count()
            next(c)
            while os.path.exists(html_out):
                html_out = f'{basename} ({str(next(c))}).{ext}'
        curr_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(curr_path, 'resources', 'suite_output.html'), 'r', encoding='utf8') as html_file:
            html_formatted = re.sub('{', '{{', html_file.read())
            html_formatted = re.sub('}', '}}', html_formatted)
            html_formatted = re.sub('html_title', '{title}', html_formatted)
            html_formatted = re.sub('widget_snippet', '{snippet}', html_formatted)
            embed_minimal_html(html_out, views=[page], title='Suite Output', template=html_formatted)
    else:
        display(page)


def _display_suite_no_widgets(summary: str,
                              unique_id: str,
                              checks_with_conditions: List[CheckResult],
                              checks_wo_conditions_display: List[CheckResult],
                              checks_w_condition_display: List[CheckResult],
                              others_table: List,
                              light_hr: str):  # pragma: no cover
    """Display results of suite in IPython without widgets."""
    bold_hr = '<hr style="background-color: black;border: 0 none;color: black;height: 1px;">'

    display_html(bold_hr + summary, raw=True)

    if checks_with_conditions:
        cond_html_table = dataframe_to_html(get_conditions_table(checks_with_conditions, unique_id, 300))
        display_html(_CONDITIONS_SUMMARY_TITLE + cond_html_table, raw=True)
    else:
        display_html(_NO_CONDITIONS_SUMMARY_TITLE, raw=True)

    outputs_h2 = f'{bold_hr}{_CHECKS_WITH_CONDITIONS_TITLE}'
    display_html(outputs_h2, raw=True)
    if checks_w_condition_display:
        for i, r in enumerate(checks_w_condition_display):
            r.show(unique_id=unique_id)
            if i < len(checks_w_condition_display) - 1:
                display_html(light_hr, raw=True)
    else:
        display_html(_NO_OUTPUT_TEXT, raw=True)

    outputs_h2 = f'{bold_hr}{_CHECKS_WITHOUT_CONDITIONS_TITLE}'
    display_html(outputs_h2, raw=True)
    if checks_wo_conditions_display:
        for i, r in enumerate(checks_wo_conditions_display):
            r.show(unique_id=unique_id)
            if i < len(checks_wo_conditions_display) - 1:
                display_html(light_hr, raw=True)
    else:
        display_html(_NO_OUTPUT_TEXT, raw=True)

    if others_table:
        others_table = pd.DataFrame(data=others_table, columns=['Check', 'Reason', 'sort'])
        others_table.sort_values(by=['sort'], inplace=True)
        others_table.drop('sort', axis=1, inplace=True)
        others_h2 = f'{bold_hr}{_CHECKS_WITHOUT_DISPLAY_TITLE}'
        others_df = dataframe_to_html(others_table.style.hide_index())
        display_html(others_h2 + others_df, raw=True)

    display_html(f'<br><a href="#summary_{unique_id}" style="font-size: 14px">Go to top</a>', raw=True)


def display_suite_result(suite_name: str, results: List[Union[CheckResult, CheckFailure]],
                         html_out=None):  # pragma: no cover
    """Display results of suite in IPython."""
    if len(results) == 0:
        display_html(f"""<h1>{suite_name}</h1><p>Suite is empty.</p>""", raw=True)
        return
    if 'google.colab' in str(get_ipython()):
        unique_id = ''
    else:
        unique_id = get_random_string()

    checks_with_conditions: List[CheckResult] = []
    checks_wo_conditions_display: List[CheckResult] = []
    checks_w_condition_display: List[CheckResult] = []
    others_table = []

    for result in results:
        if isinstance(result, CheckResult):
            if result.have_conditions():
                checks_with_conditions.append(result)
                if result.have_display():
                    checks_w_condition_display.append(result)
            elif result.have_display():
                checks_wo_conditions_display.append(result)
            if not result.have_display():
                others_table.append([result.get_header(), 'Nothing found', 2])
        elif isinstance(result, CheckFailure):
            error_types = (
                errors.DatasetValidationError,
                errors.ModelValidationError,
                errors.DeepchecksProcessError,
            )
            if isinstance(result.exception, error_types):
                msg = str(result.exception)
            else:
                msg = result.exception.__class__.__name__ + ': ' + str(result.exception)
            name = result.header
            others_table.append([name, msg, 1])
        else:
            # Should never reach here!
            raise errors.DeepchecksValueError(
                f"Expecting list of 'CheckResult'|'CheckFailure', but got {type(result)}."
            )

    checks_w_condition_display = sorted(checks_w_condition_display, key=lambda it: it.priority)

    light_hr = '<hr style="background-color: #eee;border: 0 none;color: #eee;height: 4px;">'

    icons = """
    <span style="color: green;display:inline-block">\U00002713</span> /
    <span style="color: red;display:inline-block">\U00002716</span> /
    <span style="color: orange;font-weight:bold;display:inline-block">\U00000021</span>
    """

    check_names = list(set(it.check.name() for it in results))
    prologue = (
        f"The suite is composed of various checks such as: {', '.join(check_names[:3])}, etc..."
        if len(check_names) > 3
        else f"The suite is composed of the following checks: {', '.join(check_names)}."
    )

    suite_creation_example_link = (
        'https://docs.deepchecks.com/en/stable/examples/guides/create_a_custom_suite.html'
        '?utm_source=suite_output&utm_medium=referral&utm_campaign=display_link'
    )

    # suite summary
    summ = f"""
        <h1 id="summary_{unique_id}">{suite_name}</h1>
        <p>
            {prologue}<br>
            Each check may contain conditions (which will result in pass / fail / warning, represented by {icons})
            as well as other outputs such as plots or tables.<br>
            Suites, checks and conditions can all be modified (see the
            <a href={suite_creation_example_link} target="_blank">Create a Custom Suite</a> tutorial).
        </p>
        """

    if html_out or is_widgets_enabled():
        _display_suite_widgets(summ,
                               unique_id,
                               checks_with_conditions,
                               checks_wo_conditions_display,
                               checks_w_condition_display,
                               others_table,
                               light_hr,
                               html_out)
    else:
        _display_suite_no_widgets(summ,
                                  unique_id,
                                  checks_with_conditions,
                                  checks_wo_conditions_display,
                                  checks_w_condition_display,
                                  others_table,
                                  light_hr)
