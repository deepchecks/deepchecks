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
:root {
    --deepchecks-font-color: #212529;
    --deepchecks-bg-color: white;
    --deepchecks-link-color: #106ba3;
    --deepchecks-color-dark: #d6d6d6;
    --deepchecks-color-light: #f9f9f9;
    --deepchecks-color-blue: #1975FA;
    --deepchecks-i-arrow-up: '⬆';
    --deepchecks-i-window-expand: '⤡';
    --deepchecks-i-ok: '✓';
    --deepchecks-i-error: '✖';
    --deepchecks-i-warn: '!';
    --deepchecks-i-attention: '⁈';
}

.deepchecks-table {
    max-width: 100%!important;
    overflow-x: auto!important;
}

.deepchecks-collapsible {
    position: relative;
    display: block;
    width: 100%;
    max-width: 100%;
    overflow-x: auto;
    border: 1px solid var(--deepchecks-color-dark);
    margin: 0;
    margin-top: 0.25em;
    margin-bottom: 0.25em;
}

.deepchecks-collapsible-content {
    position: relative;
    display: none;
    padding: 1em 1.5em 1em 1.5em;
}

.deepchecks-collapsible[open] > .deepchecks-collapsible-content {
    display: flex;
    flex-direction: column;
}

.deepchecks-collapsible > summary {
    display: list-item;
    background-color: var(--deepchecks-color-light);
    font-size: 1em;
    color: var(--deepchecks-font-color);
    font-weight: bold;
    padding: 10px 15px 10px 15px;
    cursor: pointer;
    user-select: none;
}

.deepchecks-collapsible[open] > summary {
    border-bottom: 1px solid var(--deepchecks-color-dark);
}

.deepchecks-tabs {
    width: 100%;
    display: flex;
    flex-direction: column;
    margin-top: 1em;
}

.deepchecks-tabs-btns {
    width: 100%;
    height: fit-content;
    display: flex;
    flex-direction: row;
}

.deepchecks-tabs-btns > button {
    margin: 0;
    background-color: var(--deepchecks-color-light);
    border: 1px solid var(--deepchecks-color-dark);
    padding: 8px 16px 8px 16px;
    cursor: pointer;
    transform: translateY(1px);
    z-index: 2;
}

.deepchecks-tabs-btns > button:focus {
    box-shadow: none;
    outline: none;
}

.deepchecks-tabs-btns > button[open] {
    background-color: white;
    border-bottom: none;
    border-top: 2px solid var(--deepchecks-color-blue);
}

.deepchecks-tabs > .deepchecks-tab {
    display: None;
}

.deepchecks-tabs > .deepchecks-tab[open] {
    display: flex;
    flex-direction: column;
    border: 1px solid var(--deepchecks-color-dark);
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
    background-color: var(--deepchecks-color-light);
    border: 1px solid var(--deepchecks-color-dark);
    border-radius: 50%;
    cursor: pointer;
    padding: 6px;
}

.deepchecks-fullscreen-btn:hover {
    opacity: 1;
}

.deepchecks-fullscreen-btn::after {
    content: var(--deepchecks-i-window-expand);
}

.deepchecks-i-expandable::after {
    content: var(--deepchecks-i-window-expand);
}

.deepchecks-i-arrow-up::after {
    content: var(--deepchecks-i-arrow-up);
}

.deepchecks-i-ok::after {
    color: green;
    font-weight: bold;
    content: var(--deepchecks-i-ok);
}

.deepchecks-i-error::after {
    color: red;
    font-weight: bold;
    content: var(--deepchecks-i-error);
}

.deepchecks-i-warn::after {
    color: orange;
    font-weight: bold;
    content: var(--deepchecks-i-warn);
}

.deepchecks-i-attention::after {
    color: firebrick;
    font-weight: bold;
    content: var(--deepchecks-i-attention);
}

.deepchecks-bold-divider {
    display: block!important;
    background-color: var(--deepchecks-color-dark)!important;
    border: 0 none!important;
    color: var(--deepchecks-color-dark)!important;
    height: 1px!important;
    width: 100%!important;
}

.deepchecks-light-divider {
    display: block!important;
    background-color: var(--deepchecks-color-light)!important;
    border: 0 none!important;
    color: var(--deepchecks-color-light)!important;
    height: 4px!important;
    width: 100%!important;
}
"""


DEEPCHECKS_HTML_PAGE_STYLE = """
%deepchecks-style

html {
  box-sizing: border-box;
  -moz-tab-size: 4;
  tab-size: 4;
}

*, ::before, ::after {
  background-repeat: no-repeat;
  box-sizing: inherit;
}

* {
    margin: 0;
    padding: 0;
}

details, main {
    display: block;
}

summary {
    display: list-item;
}

iframe {
    border-style: none;
    resize: vertical;
}

button, select {
    font: inherit;
}

progress {
    vertical-align: baseline;
}

body {
    display: flex;
    flex-direction: column;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif;
    font-size: 14px;
    line-height: 1.5;
    text-rendering: optimizeLegibility;
    word-wrap: break-word;
    color: var(--deepchecks-font-color);
    background-color: white;
    padding: 0 1rem 0 1rem;
}

@media (width >= 1000px) { 
    body {
        padding: 20px 10vw 0 10vw;
    }
}

h1, h2, h3, h4, h5, h6 {
    margin-bottom: 12px;
    margin-top: 24px;
}

h5, h6 {
    font-size: 1em;
}

p {
    margin-top: 1em;
    margin-bottom: 1em;
}

a {
  text-decoration: none;
  color: var(--deepchecks-link-color);
}

a:hover {
  text-decoration: underline;
}

a > code, a > strong {
    color: inherit;
}

table {
    font-size: 12px;
    text-indent: 0;
    border-collapse: collapse;
    margin-bottom: 10px;
    margin-bottom: 10px;
    max-width: 100%;
    overflow-x: auto;
    table-layout: fixed;
}

table caption {
    text-align: left;
}

td, th {
    padding: 6px;
    vertical-align: top;
    word-wrap: break-word;
    text-align: left;
}

th {
    font-weight: bold;
}

thead {
    border-bottom: 1px solid black;
    vertical-align: bottom;
}

tfoot {
    border-top: 1px solid black;
}

tbody tr:nth-child(odd) {
    background-color: var(--deepchecks-color-light);
}


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
