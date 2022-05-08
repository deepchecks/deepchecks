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
# pylint: disable=unused-import,import-outside-toplevel, protected-access
"""Module with common utilities routines for serialization subpackage."""
import io
import json
import os
import pkgutil
import textwrap
import typing as t
import warnings
from contextlib import contextmanager

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io._html as plotlyhtml
from ipywidgets import DOMWidget
from jsonpickle.pickler import Pickler
from pandas.io.formats.style import Styler
from plotly.io._utils import plotly_cdn_url
from plotly.offline.offline import get_plotlyjs

from deepchecks.core import check_result as check_types
from deepchecks.core import checks
from deepchecks.utils.dataframes import un_numpy
from deepchecks.utils.strings import get_ellipsis

__all__ = [
    'aggregate_conditions',
    'form_output_anchor',
    'form_check_id',
    'Html',
    'normalize_widget_style',
    'normalize_value',
    'prettify',
    'read_matplot_figures',
    'concatv_images',
    'switch_matplot_backend',
    'plotlyjs_script',
    'requirejs_script'
]


class Html:
    """Set of commonly used HTML tags."""

    bold_hr = '<hr style="background-color: black;border: 0 none;color: black;height: 1px;">'
    light_hr = '<hr style="background-color: #eee;border: 0 none;color: #eee;height: 4px;">'


def form_output_anchor(output_id: str) -> str:
    """Form unique output anchor."""
    return f'summary_{output_id}'


def form_check_id(check_name: str, output_id: str) -> str:
    """Form check instance unique identifier."""
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


def prettify(data: t.Any, indent: int = 3) -> str:
    """Prettify data."""
    return json.dumps(data, indent=indent, default=repr)


def normalize_value(value: object) -> t.Any:
    """Take an object and return a JSON-safe representation of it.

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
    check_results: t.Union['check_types.CheckResult', t.List['check_types.CheckResult']],
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
    check_results = (
        [check_results]
        if isinstance(check_results, check_types.CheckResult)
        else check_results
    )

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


def requirejs_script(connected: bool = True):
    """Return requirejs script tag.

    Parameters
    ----------
    connected : bool, default True

    Returns
    -------
    str
    """
    if connected is True:
        return textwrap.dedent("""
            <script
                src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"
                integrity="sha512-c3Nl8+7g4LMSTdrm621y7kf9v3SDPnhxLNhcjFJbKECVnmZHTdo+IRO05sNLTH/D3vA6u1X32ehoLC7WFVdheg=="
                crossorigin="anonymous"
                referrerpolicy="no-referrer">
            </script>
        """)
    else:
        path = os.path.join('core', 'resources', 'requirejs.min.js')
        js = pkgutil.get_data('deepchecks', path).decode('utf-8')
        return f'<script>{js}</script>'


def plotlyjs_script(connected: bool = True) -> str:
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
            {win_config}
            {mathjax_config}
            <script type="text/javascript">
                if (typeof require !== 'undefined') {{
                    require.undef("plotly");
                    requirejs.config({{
                        paths: {{'plotly': ['{plotly_cdn}']}}
                    }});
                    require(
                        ['plotly'],
                        function(Plotly) {{
                            window._Plotly = Plotly;
                            window.Plotly = Plotly;
                            console.log('Loaded plotly successfully');
                        }},
                        function() {{console.log('Failed to load plotly')}}
                    );
                }} else {{
                    console.log('requirejs is not present');
                }}
            </script>
        """)
        return script.format(
            win_config=plotlyhtml._window_plotly_config,
            mathjax_config=plotlyhtml._mathjax_config,
            plotly_cdn=plotly_cdn_url().rstrip('.js'),
        )
    else:
        # If not connected then we embed a copy of the plotly.js library
        script = textwrap.dedent("""
            {win_config}
            {mathjax_config}
            <script type="text/javascript">
                if (typeof require !== 'undefined') {{
                    require.undef("plotly");
                    define('plotly', function(require, exports, module) {{
                        {script}
                    }});
                    require(
                        ['plotly'],
                        function(Plotly) {{
                            window._Plotly = Plotly;
                            window.Plotly = Plotly;
                            console.log('Loaded plotly successfully');
                        }},
                        function() {{console.log('Failed to load plotly')}}
                    );
                }} else {{
                    console.log('requirejs is not present');
                }}
            </script>
        """)
        return script.format(
            script=get_plotlyjs(),
            win_config=plotlyhtml._window_plotly_config,
            mathjax_config=plotlyhtml._mathjax_config,
        )


def read_matplot_figures() -> t.List[io.BytesIO]:
    """Return all active matplot figures."""
    output = []
    figures = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figures:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        output.append(buffer)
        fig.clear()
        plt.close(fig)
    return output


@contextmanager
def switch_matplot_backend(backend: str = 'agg'):
    """Switch matplot backend."""
    previous = matplotlib.get_backend()
    plt.switch_backend(backend)
    yield
    plt.switch_backend(previous)


def concatv_images(images, gap=10):
    """Concatenate a list of images vertically.

    Parameters
    ----------
    images : List[PIL.Image.Image]
        list of images
    gap : int, default 10
        gap between images

    Returns
    -------
    PIL.Image.Image
    """
    try:
        import PIL.Image as pilimage
    except ImportError as e:
        raise ImportError(
            'concatv_images function requires the PIL package. '
            'To get it, run "pip install pillow".'
        ) from e
    else:
        assert isinstance(images, list) and len(images) != 0
        assert isinstance(gap, int) and gap >= 0

        if len(images) == 1:
            return t.cast(pilimage.Image, images[0]).copy()

        max_width = max(it.width for it in images)
        height = sum(it.height for it in images)
        dst = pilimage.new(
            t.cast(pilimage.Image, images[0]).mode,  # type: ignore
            (max_width, height)
        )

        position = 0

        for img in images:
            dst.paste(img, (0, position))
            position = position + img.height + gap

        return dst
