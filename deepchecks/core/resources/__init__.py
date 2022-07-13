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
"""Package for common static resources."""
import os
import pkgutil
import textwrap

from ipywidgets.embed import __html_manager_version__

__all__ = ['DEEPCHECKS_STYLE', 'DEEPCHECKS_HTML_PAGE_STYLE']


DEEPCHECKS_STYLE = """

.deepchecks {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'!important;
    font-size: 14px!important;
    line-height: 1.5!important;
    color: #212529!important;
    text-align: left!important;
    background-color: white!important;
}
.deepchecks h1,
.deepchecks h2,
.deepchecks h3,
.deepchecks h4,
.deepchecks h5,
.deepchecks h6 {
    font-style: normal!important;
    font-weight: 600!important;
    text-transform: none!important;
    text-decoration: none!important;
    text-align: left!important;
    margin: revert!important;
    margin-top: revert!important;
    margin-bottom: revert!important;
}
.deepchecks p {
    font-size: revert!important;
    margin: revert!important;
    margin-top: revert!important;
    margin-bottom: revert!important;
    text-align: justify!important;
}
.deepchecks a:link {
    font-size: revert!important;
    color: #106ba3!important;
    text-decoration: none!important;
}
.deepchecks a:link:hover {
    text-decoration: underline!important;
    cursor: pointer!important;
}
.deepchecks table {
    display: block!important;
    width: fit-content!important;
    max-width: 100%!important;
    overflow-x: auto!important;
    font-size: 12px!important;
    text-align: left!important;
}
.deepchecks tr,
.deepchecks th,
.deepchecks td {
    text-align: left!important;
}
.deepchecks iframe {
    resize: vertical!important;
}

details.deepchecks {
    position: relative;
    border: 1px solid #d6d6d6;
    margin-bottom: 0.25em;
}
details.deepchecks > div {
    position: relative;
    display: none;
    flex-direction: column;
    padding: 1em 1.5em 1em 1.5em;
}
details.deepchecks[open] > div {
    display: flex;
}
details.deepchecks > summary {
    display: list-item;
    background-color: #f9f9f9;
    font-weight: bold;
    padding: 10px 15px 10px 15px;
    cursor: pointer;
    user-select: none;
}
details.deepchecks[open] > summary {
    border-bottom: 1px solid #d6d6d6;
}
div.deepchecks-tabs {
    width: 100%;
    display: flex;
    flex-direction: column;
    margin-top: 1em;
}
div.deepchecks-tabs-btns {
    width: 100%;
    height: fit-content;
    display: flex;
    flex-direction: row;
}
div.deepchecks-tabs-btns > button {
    margin: 0;
    background-color: #f9f9f9;
    border: 1px solid #d6d6d6;
    padding: 8px 16px 8px 16px;
    cursor: pointer;
    transform: translateY(1px);
    z-index: 2;
}
div.deepchecks-tabs-btns > button:focus {
    box-shadow: none;
    outline: none;
}
div.deepchecks-tabs-btns > button[open] {
    background-color: white;
    border-bottom: none;
    border-top: 2px solid #1975FA;
}
div.deepchecks-tabs > div.deepchecks-tab {
    display: None;
}
div.deepchecks-tabs > div.deepchecks-tab[open] {
    display: flex;
    flex-direction: column;
    border: 1px solid #d6d6d6;
    padding: 1em;
    z-index: 1;
}
.deepchecks-alert {
    border: 1px solid transparent;
    border-radius: 0.25em;
    padding: 0.5em 1em 0.5em 1em;
    margin-top: 0.5em;
    margin-bottom: 0.5em;
}
.deepchecks-alert-error {
    color: #721c24;
    background-color: #f8d7da;
    border-color: #f5c6cb;
}
.deepchecks-alert-warn {
    color: #856404;
    background-color: #fff3cd;
    border-color: #ffeeba;
}
.deepchecks-alert-info {
    color: #004085;
    background-color: #cce5ff;
    border-color: #b8daf;
}
.deepchecks-fullscreen-btn {
    position: absolute;
    bottom: 30px;
    right: 60px;
    opacity: 0.4;
    font-size: 32px;
    font-weight: 600;
    line-height: 1;
    background-color: #f9f9f9;
    border: 1px solid #d6d6d6;
    border-radius: 50%;
    cursor: pointer;
    padding: 6px;
    text-align: center;
    vertical-align: middle;
}
.deepchecks-fullscreen-btn:hover {
    opacity: 1;
}
.deepchecks-fullscreen-btn::after {
    content: '⤡';
}
"""


DEEPCHECKS_HTML_PAGE_STYLE = """
body.deepchecks {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
    font-size: 14px;
    line-height: 1.5;
    color: #212529;
    text-align: left;
    margin: auto;
    background-color: white;
    padding: 1rem 1rem 0 1rem;
}
.deepchecks table {
    border: none;
    border-collapse: collapse;
    border-spacing: 0;
    color: black;
    max-width: fit-content;
}
.deepchecks thead {
    border-bottom: 1px solid black;
    vertical-align: bottom;
}
.deepchecks tr,
.deepchecks th,
.deepchecks td {
    text-align: left;
    vertical-align: middle;
    padding: 0.5em 0.5em;
    line-height: normal;
    white-space: normal;
    max-width: none;
    border: none;
}
.deepchecks th {
    font-weight: bold;
}
.deepchecks tbody tr:nth-child(odd) {
    background: white;
}
.deepchecks tbody tr:nth-child(even) {
    background: #f5f5f5;
}
.deepchecks tbody tbody tr:hover {
    background: rgba(66, 165, 245, 0.2);
}
%deepchecks-style
""".replace('%deepchecks-style', DEEPCHECKS_STYLE)


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
