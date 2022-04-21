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
"""Module with common utilities routines for serialization subpackage."""
import typing as t
import warnings
import json
import textwrap

import pandas as pd
import numpy as np
from ipywidgets import DOMWidget
from jsonpickle.pickler import Pickler
from pandas.io.formats.style import Styler
from plotly.io._utils import plotly_cdn_url
from plotly.offline.offline import get_plotlyjs

from deepchecks.utils.strings import get_ellipsis
from deepchecks.core.check_result import CheckResult
from deepchecks.core.checks import BaseCheck
from deepchecks.utils.dataframes import un_numpy


__all__ = [
    'aggregate_conditions',
    'form_output_anchor',
    'form_check_id',
    'Html',
    'normalize_widget_style',
    'normalize_value',
    'pretify',
    'plotly_activation_script'
]


# class CustomNotebookRenderer(plotly_renderes.NotebookRenderer):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.connected = False
#         self._is_activated = False

#     def activate(self):
#         if self._is_activated is False:
#             super().activate()
#             self._is_activated = True

#     @property
#     def is_plotly_activated(self):
#         return self._is_activated


class Html:
    """Set of commonly used HTML tags."""

    bold_hr = '<hr style="background-color: black;border: 0 none;color: black;height: 1px;">'
    light_hr = '<hr style="background-color: #eee;border: 0 none;color: #eee;height: 4px;">'


def form_output_anchor(output_id: str) -> str:
    """Form unique output anchor."""
    return f'summary_{output_id}'


def form_check_id(check: BaseCheck, output_id: str) -> str:
    """Form check instance unique identifier."""
    check_name = type(check).__name__
    return f'{check_name}_{output_id}'


TDOMWidget = t.TypeVar('TDOMWidget', bound=DOMWidget)


def normalize_widget_style(w: TDOMWidget) -> TDOMWidget:
    """Add additional style classes to the widget."""
    return (
        w
        .add_class('rendered_html')
        .add_class('jp-RenderedHTMLCommon')
        .add_class('jp-RenderedHTML')
        .add_class('jp-OutputArea-output')
    )


def pretify(data: t.Any, indent: int = 3) -> str:
    """Pretify data."""
    default = lambda it: repr(it)
    return json.dumps(data, indent=indent, default=default)


def normalize_value(value: object) -> t.Any:
    """Takes an object and returns a JSON-safe representation of it.

    Parameters
    ----------
    value : object
        value to normilize

    Returns
    -------
    Any of the basic builtin datatypes
    """
    if isinstance(value, pd.DataFrame):
        return value.to_json(orient='records')
    elif isinstance(value, Styler):
        return value.data.to_json(orient='records')
    elif isinstance(value, (np.generic, np.ndarray)):
        return un_numpy(value)
    else:
        return Pickler(unpicklable=False).flatten(value)


def aggregate_conditions(
    check_results: t.Union['CheckResult', t.List['CheckResult']],
    max_info_len: int = 3000,
    include_icon: bool = True,
    include_check_name: bool = False,
    output_id: t.Optional[str] = None,
) -> pd.DataFrame:
    """Return the conditions table as DataFrame.

    Parameters
    ----------
    check_results : Union['CheckResult', List['CheckResult']]
        check results to show conditions of.
    max_info_len : int
        max length of the additional info.
    include_icon : bool , default: True
        if to show the html condition result icon or the enum
    include_check_name : bool, default False
        whether to include check name into dataframe or not
    output_id : str
        the unique id to append for the check names to create links (won't create links if None/empty).

    Returns
    -------
    pd.Dataframe:
        the condition table.
    """
    check_results = [check_results] if isinstance(check_results, CheckResult) else check_results
    data = []

    for check_result in check_results:
        for cond_result in check_result.conditions_results:
            priority = cond_result.priority
            icon = cond_result.get_icon() if include_icon else cond_result.category.value
            check_header = check_result.get_header()

            if output_id:
                link = f'<a href=#{check_result.get_check_id(output_id)}>{check_header}</a>'
            else:
                link = check_header
                # if it has no display show on bottom for the category (lower priority)
                priority += 0.1

            data.append([
                icon, link, cond_result.name, cond_result.details, priority
            ])

    df = pd.DataFrame(
        data=data,
        columns=['Status', 'Check', 'Condition', 'More Info', 'sort']
    )

    df.sort_values(by=['sort'], inplace=True)
    df.drop('sort', axis=1, inplace=True)

    if include_check_name is False:
        df.drop('Check', axis=1, inplace=True)

    df['More Info'] = df['More Info'].map(lambda x: get_ellipsis(x, max_info_len))

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        return df.style.hide_index()


REQUIREJS_CDN = """
<!-- Load require.js. Delete this if your page already loads require.js -->
<script
    src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
    integrity="sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA="
    crossorigin="anonymous">
</script>
"""


# HTML
# Build script to set global PlotlyConfig object. This must execute before
# plotly.js is loaded.
_window_plotly_config = """window.PlotlyConfig = {MathJaxConfig: 'local'};"""
_mathjax_config = """if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}"""


def plotly_activation_script(connected: bool = True) -> str:
    """Return plotly activation script in the requirejs enviroment.

    Parameters
    ----------
    connected : bool, default True

    Returns
    -------
    str
    """
    if connected is True:
        # Connected so we configure requirejs with the plotly CDN
        script = textwrap.dedent("""
            <script type="text/javascript">
                if (typeof require !== 'undefined') {{
                    require(['plotly'], function () {{}}, function (error) {{
                        {win_config}
                        {mathjax_config}
                        require.undef("plotly");
                        requirejs.config({{
                            paths: {{'plotly': ['{plotly_cdn}']}}
                        }});
                        require(
                            ['plotly'],
                            function(Plotly) {{window._Plotly = Plotly;}},
                            function() {{console.log('Failed to load plotly')}}
                        );
                    }});
                }} else {{
                    console.log('requirejs is not present');
                }}
            </script>
        """)
        return script.format(
            win_config=_window_plotly_config,
            mathjax_config=_mathjax_config,
            plotly_cdn=plotly_cdn_url().rstrip(".js"),
        )
    else:
        # If not connected then we embed a copy of the plotly.js
        # library in the notebook
        script = textwrap.dedent("""
            <script type="text/javascript">
                if (typeof require !== 'undefined') {{
                    require(['plotly'], function () {{}}, function (error) {{
                        {win_config}
                        {mathjax_config}
                        require.undef("plotly");
                        define('plotly', function(require, exports, module) {{
                            {script}
                        }});
                        require(
                            ['plotly'],
                            function(Plotly) {{window._Plotly = Plotly;}},
                            function() {{console.log('Failed to load plotly')}}
                        );
                    }})
                }} else {{
                    console.log('requirejs is not present');
                }}
            </script>
        """)
        return script.format(
            script=get_plotlyjs(),
            win_config=_window_plotly_config,
            mathjax_config=_mathjax_config,
        )
