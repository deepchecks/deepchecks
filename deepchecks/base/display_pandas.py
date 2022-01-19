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
"""Handle displays of pandas objects."""
from typing import List, Union
import warnings

import pandas as pd
from pandas.io.formats.style import Styler

from deepchecks.utils.strings import get_docs_summary, get_ellipsis

from . import check  # pylint: disable=unused-import


__all__ = ['dataframe_to_html', 'get_conditions_table']


def dataframe_to_html(df: Union[pd.DataFrame, Styler]):
    """Convert dataframe to html.

    Args:
        df (Union[pd.DataFrame, Styler]): Dataframe to convert to html
    Returns:
        pd.DataFrame:
            dataframe with styling
    """
    try:
        if isinstance(df, pd.DataFrame):
            df_styler = df.style
        else:
            df_styler = df
        # Using deprecated pandas method so hiding the warning
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            df_styler.set_precision(2)

        table_css_props = [
            ('text-align', 'left'),  # Align everything to the left
            ('white-space', 'pre-wrap')  # Define how to handle white space characters (like \n)
        ]
        df_styler.set_table_styles([dict(selector='table,thead,tbody,th,td', props=table_css_props)])
        return df_styler.render()
    # Because of MLC-154. Dataframe with Multi-index or non unique indices does not have a style
    # attribute, hence we need to display as a regular pd html format.
    except ValueError:
        return df.to_html()


def get_conditions_table(check_results: Union['check.CheckResult', List['check.CheckResult']],
                         unique_id=None, max_info_len: int = 3000):
    """Return the conditions table as DataFrame.

    Args:
        check_results (Union['CheckResult', List['CheckResult']]): check results to show conditions of.
        unique_id (str): the unique id to append for the check names to create links
                              (won't create links if None/empty).
        max_info_len (int): max length of the additional info.
    Returns:
        pd.Dataframe:
            the condition table.
    """
    if not isinstance(check_results, List):
        show_check_column = False
        check_results = [check_results]
    else:
        show_check_column = True

    table = []
    for check_result in check_results:
        for cond_result in check_result.conditions_results:
            sort_value = cond_result.priority
            icon = cond_result.get_icon()
            check_header = check_result.get_header()
            if unique_id and check_result.have_display():
                check_id = f'{check_result.check.__class__.__name__}_{unique_id}'
                link = f'<a href=#{check_id}>{check_header}</a>'
            else:
                link = check_header
                sort_value = 1 if sort_value == 1 else 5  # if it failed but has no display still show on top
            table.append([icon, link, cond_result.name,
                         cond_result.details, sort_value])

    conditions_table = pd.DataFrame(data=table,
                                    columns=['Status', 'Check', 'Condition', 'More Info', 'sort'])
    conditions_table.sort_values(by=['sort'], inplace=True)
    conditions_table.drop('sort', axis=1, inplace=True)
    if show_check_column is False:
        conditions_table.drop('Check', axis=1, inplace=True)
    conditions_table['More Info'] = conditions_table['More Info'].map(lambda x: get_ellipsis(x, max_info_len))
    return conditions_table.style.hide_index()


def get_result_navigation_display(check_results: Union['check.CheckResult', List['check.CheckResult']],
                                  unique_id: str):
    """Display the results as a table with links for navigation.

    Args:
        check_results (Union['CheckResult', List['CheckResult']]): check results to show navigation for.
        unique_id (str): the unique id to append for the check names to create links.
    Returns:
        str:
            html representation of the navigation table.
    """
    table = []
    for check_result in check_results:
        if check_result.have_display():
            check_header = check_result.get_header()
            check_id = f'{check_result.check.__class__.__name__}_{unique_id}'
            link = f'<a href=#{check_id}>{check_header}</a>'
            summary = get_docs_summary(check_result.check)
            table.append([link, summary])

    nav_table = pd.DataFrame(data=table,
                             columns=['Check', 'Summary'])
    return dataframe_to_html(nav_table.style.hide_index())
