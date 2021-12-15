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

# pylint: disable=protected-access
import sys
import tqdm
import pandas as pd

from IPython.core.display import display_html

from deepchecks.base.check import CheckResult, CheckFailure, ConditionResult
from deepchecks.base.display_pandas import dataframe_to_html, display_conditions_table


__all__ = ['display_suite_result', 'ProgressBar']


class ProgressBar:
    """Progress bar for display while running suite."""

    def __init__(self, name, length):
        """Initialize progress bar."""
        self.pbar = tqdm.tqdm(total=length,desc=name, unit='Check', \
                              leave=False, file=sys.stdout, \
                              bar_format=f'{{l_bar}}{{bar:{length}}}{{r_bar}}')

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


def display_suite_result(suite_name: str, results: List[Union[CheckResult, CheckFailure]]):
    """Display results of suite in IPython."""
    conditions_table = []
    display_table = []
    others_table = []
    for result in results:
        if isinstance(result, CheckResult):
            if result.have_conditions():
                ConditionResult.append_to_conditions_table(result, conditions_table)
            if result.have_display():
                display_table.append(result)
            else:
                others_table.append([result.get_header(), 'Nothing found', 2])
        elif isinstance(result, CheckFailure):
            msg = result.exception.__class__.__name__ + ': ' + str(result.exception)
            name = result.check.name()
            others_table.append([name, msg, 1])

    light_hr = '<hr style="background-color: #eee;border: 0 none;color: #eee;height: 1px;">'
    bold_hr = '<hr style="background-color: black;border: 0 none;color: black;height: 1px;">'
    icons = """
    <span style="color: green;display:inline-block">\U00002713</span> /
    <span style="color: red;display:inline-block">\U00002716</span> /
    <span style="color: orange;font-weight:bold;display:inline-block">\U00000021</span>
    """
    html = f"""
    <h1>{suite_name}</h1>
    <p>The suite is composed of various checks such as: {get_first_3(results)}, etc...<br>
    Each check may contain conditions (which results in {icons}), as well as other outputs such as plots or tables.<br>
    Suites, checks and conditions can all be modified (see tutorial [link]).</p>
    {bold_hr}<h2>Conditions Summary</h2>
    """
    display_html(html, raw=True)
    if conditions_table:
        display_conditions_table(conditions_table)
    else:
        display_html('<p>No conditions defined on checks in the suite.</p>', raw=True)

    display_html(f'{bold_hr}<h2>Additional Outputs</h2>', raw=True)
    if display_table:
        for i, r in enumerate(display_table):
            r._ipython_display_()
            if i < len(display_table) - 1:
                display_html(light_hr, raw=True)
    else:
        display_html('<p>No outputs to show.</p>', raw=True)

    if others_table:
        others_table = pd.DataFrame(data=others_table, columns=['Check', 'Reason', 'sort'])
        others_table.sort_values(by=['sort'], inplace=True)
        others_table.drop('sort', axis=1, inplace=True)
        html = f"""{bold_hr}
        <h2>Other Checks That Weren't Displayed</h2>
        {dataframe_to_html(others_table.style.hide_index())}
        """
        display_html(html, raw=True)


def get_first_3(results: List[Union[CheckResult, CheckFailure]]):
    first_3 = []
    i = 0
    while len(first_3) < 3 and i < len(results):
        curr = results[i]
        curr_name = curr.check.name()
        if curr_name not in first_3:
            first_3.append(curr_name)
        i += 1
    return ', '.join(first_3)
