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
from tqdm.notebook import tqdm as tqdm_notebook
import pandas as pd
from IPython.core.display import display_html
from IPython import get_ipython

from deepchecks import errors
from deepchecks.utils.ipython import is_widgets_enabled
from deepchecks.utils.strings import get_random_string
from deepchecks.base.check import CheckResult, CheckFailure
from deepchecks.base.display_pandas import dataframe_to_html, display_conditions_table


__all__ = ['display_suite_result', 'ProgressBar']


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


def display_suite_result(suite_name: str, results: List[Union[CheckResult, CheckFailure]]):
    """Display results of suite in IPython."""
    if len(results) == 0:
        display_html(f"""<h1>{suite_name}</h1><p>Suite is empty.</p>""", raw=True)
        return
    if 'google.colab' in str(get_ipython()):
        unique_id = ''
    else:
        unique_id = get_random_string()
    checks_with_conditions = []
    display_table: List[CheckResult] = []
    others_table = []

    for result in results:
        if isinstance(result, CheckResult):
            if result.have_conditions():
                checks_with_conditions.append(result)
            if result.have_display():
                display_table.append(result)
            else:
                others_table.append([result.get_header(), 'Nothing found', 2])
        elif isinstance(result, CheckFailure):
            msg = result.exception.__class__.__name__ + ': ' + str(result.exception)
            name = result.check.name()
            others_table.append([name, msg, 1])
        else:
            # Should never reach here!
            raise errors.DeepchecksValueError(
                f"Expecting list of 'CheckResult'|'CheckFailure', but got {type(result)}."
            )

    display_table = sorted(display_table, key=lambda it: it.priority)

    light_hr = '<hr style="background-color: #eee;border: 0 none;color: #eee;height: 1px;">'
    bold_hr = '<hr style="background-color: black;border: 0 none;color: black;height: 1px;">'

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

    suite_creation_example_link = 'https://docs.deepchecks.com/en/stable/examples/guides/create_a_custom_suite.html'

    display_html(
        f"""
        <h1 id="summary_{unique_id}">{suite_name}</h1>
        <p>
            {prologue}<br>
            Each check may contain conditions (which will result in pass / fail / warning, represented by {icons})
            as well as other outputs such as plots or tables.<br>
            Suites, checks and conditions can all be modified (see the
            <a href={suite_creation_example_link}>Create a Custom Suite</a> tutorial).
        </p>
        {bold_hr}
        <h2>Conditions Summary</h2>
        """,
        raw=True
    )

    if checks_with_conditions:
        display_conditions_table(checks_with_conditions, unique_id)
    else:
        display_html('<p>No conditions defined on checks in the suite.</p>', raw=True)

    display_html(f'{bold_hr}<h2>Additional Outputs</h2>', raw=True)
    if display_table:
        for i, r in enumerate(display_table):
            r.show(show_conditions=False, unique_id=unique_id)
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

    display_html(f'<br><a href="#summary_{unique_id}" style="font-size: 14px">Go to top</a>', raw=True)
