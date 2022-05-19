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
# pylint: disable=assignment-from-none
"""Utils module containing useful global functions."""
import contextlib
import io
import os
import re
import subprocess
import sys
import typing as t
from functools import lru_cache

import tqdm
from IPython import get_ipython
from IPython.display import display
from tqdm.notebook import tqdm as tqdm_notebook

__all__ = [
    'is_notebook',
    'is_widgets_enabled',
    'is_headless',
    'create_progress_bar',
    'create_dummy_progress_bar',
    'is_colab_env',
    'is_kaggle_env',
    'is_widgets_use_possible',
]


@lru_cache(maxsize=None)
def is_notebook() -> bool:
    """Check if we're in an interactive context (Notebook, GUI support) or terminal-based.

    Returns
    -------
    bool
        True if we are in a notebook context, False otherwise
    """
    try:
        shell = get_ipython()
        return hasattr(shell, 'config')
    except NameError:
        return False  # Probably standard Python interpreter


@lru_cache(maxsize=None)
def is_headless() -> bool:
    """Check if the system can support GUI.

    Returns
    -------
    bool
        True if we cannot support GUI, False otherwise
    """
    # pylint: disable=import-outside-toplevel
    try:
        import Tkinter as tk
    except ImportError:
        try:
            import tkinter as tk
        except ImportError:
            return True
    try:
        root = tk.Tk()
    except tk.TclError:
        return True
    root.destroy()
    return False


@lru_cache(maxsize=None)
def is_widgets_enabled() -> bool:
    """Check if we're running in jupyter and having jupyter widgets extension enabled."""
    # TODO:
    # this is not the right way to verify whether widgets are enabled or not:
    #  - there is always a possibility that a user had started the jupyter server
    #    with not default config path
    #  - there are two extension types:
    #      + classical notebook extensions (managed by 'jupyter nbextension');
    #      + jupyterlab extensions (managed by 'jupyter labextension');
    if not is_notebook():
        return False
    else:
        # Test if widgets extension are in list
        try:
            # The same widget can appear multiple times from different config locations, than if there are both
            # disabled and enabled, regard it as disabled
            output = subprocess.getoutput('jupyter nbextension list').split('\n')
            disabled_regex = re.compile(r'\s*(jupyter-js-widgets/extension).*(disabled).*')
            enabled_regex = re.compile(r'\s*(jupyter-js-widgets/extension).*(enabled).*')
            found_disabled = any((disabled_regex.match(s) for s in output))
            found_enabled = any((enabled_regex.match(s) for s in output))
            return not found_disabled and found_enabled
        except Exception:  # pylint: disable=broad-except
            return False


@lru_cache(maxsize=None)
def is_colab_env() -> bool:
    """Check if we are in the google colab enviroment."""
    return 'google.colab' in str(get_ipython())


@lru_cache(maxsize=None)
def is_kaggle_env() -> bool:
    """Check if we are in the kaggle enviroment."""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None


@lru_cache(maxsize=None)
def is_widgets_use_possible() -> bool:
    """Verify if widgets use is possible within the current environment."""
    # NOTE:
    # - google colab has no support for widgets but good support for viewing html pages in the output
    # - can't display plotly widgets in kaggle notebooks
    return (
        is_widgets_enabled()
        and not is_colab_env()
        and not is_kaggle_env()
    )


def create_progress_bar(
    iterable: t.Sequence[t.Any],
    name: str,
    unit: str,
):
    """Create a tqdm progress bar instance."""
    kwargs = {
        'iterable': iterable,
        'desc': name,
        'unit': f' {unit}',
        'leave': False,
    }

    iterlen = len(iterable)
    barlen = iterlen if iterlen > 5 else 5

    if is_widgets_enabled():
        return tqdm_notebook(
            **kwargs,
            colour='#9d60fb',
            file=sys.stdout
        )

    elif is_notebook():

        class PB(tqdm.tqdm):
            """Custom progress bar."""

            def __init__(self, **kwargs):
                self.display_handler = display({'text/plain': ''}, raw=True, display_id=True)
                kwargs['file'] = io.StringIO()
                super().__init__(**kwargs)

            def refresh(self, nolock=False, lock_args=None):
                value = super().refresh(nolock, lock_args)
                self.display_handler.update({'text/plain': self.fp.getvalue()}, raw=True)
                self.fp.seek(0)
                return value

            def close(self, *args, **kwargs):
                value = super().close(*args, **kwargs)
                self.display_handler.update({'text/plain': ''}, raw=True)
                self.fp.seek(0)
                return value

        return PB(
            **kwargs,
            bar_format='{{desc}}:\n|{{bar:{0}}}{{r_bar}}'.format(barlen),  # pylint: disable=consider-using-f-string
        )

    else:
        return tqdm.tqdm(
            **kwargs,
            bar_format='{{desc}}:\n|{{bar:{0}}}{{r_bar}}'.format(barlen),  # pylint: disable=consider-using-f-string
        )


@contextlib.contextmanager
def create_dummy_progress_bar(name: str, unit: str):
    """Create a tqdm progress bar instance that has only one step."""
    for _ in create_progress_bar(
        list(range(1)),
        name=name,
        unit=unit
    ):
        yield


# TODO:
# NOTE:
#   take a look at the 'is_widgets_enabled' function
#   to understand why this code below is needed

# class JupyterServerInfo(t.NamedTuple):
#     url: str
#     directory: str


# class JupyterLabExtensionInfo(t.TypedDict):
#     name: str
#     description: str
#     url: str
#     enabled: bool
#     core: bool
#     latest_version: str
#     installed_version: str
#     status: str


# class NotebookExtensionsInfo(t.TypedDict):
#     load_extensions: t.Dict[str, bool]  # name of extension -> is enabled flag


# def get_jupyter_server_info() -> t.List[JupyterServerInfo]:
#     try:
#         output = subprocess.getoutput('jupyter server list').split('\n')
#         return [
#             JupyterServerInfo(*list(map(str.strip, it.split('::'))))
#             for it in output[1:]
#         ]
#     except BaseException:
#         return []


# def get_jupyterlab_extensions_config() -> t.List[t.Tuple[
#     JupyterServerInfo,
#     t.List[JupyterLabExtensionInfo]
# ]]:
#     output = []

#     for server in get_jupyter_server_info():
#         urlobj = urlparse(server.url)
#         url = '{}://{}/lab/api/extensions?token={}'.format(
#             urlobj.scheme,
#             urlobj.netloc,
#             _extract_jupyter_token(urlobj.query)
#         )
#         try:
#             with urllib.request.urlopen(url) as f:
#                 output.append((server, json.load(f)))
#         except:
#             pass

#     return output


# def get_notebooks_extensions_config() -> t.List[t.Tuple[
#     JupyterServerInfo,
#     NotebookExtensionsInfo
# ]]:
#     output = []

#     for server in get_jupyter_server_info():
#         urlobj = urlparse(server.url)
#         url = '{}://{}/api/config/notebook?token={}'.format(
#             urlobj.scheme,
#             urlobj.netloc,
#             _extract_jupyter_token(urlobj.query)
#         )
#         try:
#             with urllib.request.urlopen(url) as f:
#                 output.append((server, json.load(f)))
#         except BaseException:
#             pass

#     return output


# def _extract_jupyter_token(url) -> str:
#     query = parse_qs(url)
#     token = (query.get('token') or [])
#     return (token[0] if len(token) > 0 else '')


# def is_widgets_enabled() -> bool:
#     lab_config = get_jupyterlab_extensions_config()
#     notebook_config = get_notebooks_extensions_config()

#     is_lab_extension_enabled = False
#     is_notebook_extension_enabled = False

#     for _, extensions_list in lab_config:
#         is_widgets_extension_enabled = any([
#             config
#             for config in extensions_list
#             if (
#                 config['name'] == '@jupyter-widgets/jupyterlab-manager'
#                 and config['enabled']
#             )
#         ])
#         if is_widgets_extension_enabled is True:
#             is_lab_extension_enabled = True
#             break

#     for _, config in notebook_config:
#         extensions = config.get('load_extensions') or {}
#         if extensions.get('jupyter-js-widgets/extension') is True:
#             is_notebook_extension_enabled = True
#             break

#     if len(lab_config) > 1 or len(notebook_config) > 1:
#         warnings.warn('')  # TODO:
#     elif is_lab_extension_enabled is False or is_notebook_extension_enabled:
#         warnings.warn('')  # TODO:

#     return is_lab_extension_enabled or is_notebook_extension_enabled
