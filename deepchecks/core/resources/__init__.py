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
"""Package for static assets."""
import os
import pkgutil
import textwrap

from ipywidgets.embed import __html_manager_version__

__all__ = ['requirejs_script', 'widgets_script', 'suite_template', 'jupyterlab_plotly_script']


def requirejs_script(connected: bool = True):
    """Return requirejs script.

    Parameters
    ----------
    connected : bool, default True
        whether to use CDN or not

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
        js = pkgutil.get_data('deepchecks', path)
        if js is None:
            raise RuntimeError('Did not find requirejs asset.')
        js = js.decode('utf-8')
        return f'<script type="text/javascript">{js}</script>'


def widgets_script(connected: bool = True, amd_module: bool = False) -> str:
    """Return ipywidgets javascript library.

    Parameters
    ----------
    connected : bool, default True
        whether to use CDN or not
    amd_module : bool, default False
        whether to use requirejs compatiable module or not

    Returns
    -------
    str
    """
    if connected is True:
        url = (
            f'https://unpkg.com/@jupyter-widgets/html-manager@{__html_manager_version__}/dist/embed-amd.js'
            if amd_module is True
            else f'https://unpkg.com/@jupyter-widgets/html-manager@{__html_manager_version__}/dist/embed.js'
        )
        return f'<script src="{url}" crossorigin="anonymous"></script>'
    else:
        asset_name = 'widgets-embed-amd.js' if amd_module is True else 'widgets-embed.js'
        path = os.path.join('core', 'resources', asset_name)
        js = pkgutil.get_data('deepchecks', path)

        if js is None:
            raise RuntimeError('Did not find widgets javascript assets')

        js = js.decode('utf-8')
        return f'<script type="text/javascript">{js}</script>'


def jupyterlab_plotly_script(connected: bool = True) -> str:
    """Return jupyterlab-plotly javascript library.

    Parameters
    ----------
    connected : bool, default True
        whether to use CDN or not

    Returns
    -------
    str
    """
    if connected is True:
        url = 'https://unpkg.com/jupyterlab-plotly@^5.5.0/dist/index.js'
        return f'<script type="text/javascript" src="{url}" async></script>'
    else:
        path = os.path.join('core', 'resources', 'jupyterlab-plotly.js')
        js = pkgutil.get_data('deepchecks', path)

        if js is None:
            raise RuntimeError('Did not find jupyterlab-plotly javascript assets')

        js = js.decode('utf-8')
        return f'<script type="text/javascript">{js}</script>'


def suite_template(full_html: bool = True) -> str:
    """Get suite template."""
    asset_name = 'suite-template-full.html' if full_html else 'suite-template-full.html'
    path = os.path.join('core', 'resources', asset_name)
    template = pkgutil.get_data('deepchecks', path)

    if template is None:
        raise RuntimeError('Did not find suite template asset')

    return template.decode('utf-8')
