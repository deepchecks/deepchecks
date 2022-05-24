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
import io
import os
import re
import subprocess
import sys
import typing as t
from functools import lru_cache

import tqdm
from ipykernel.zmqshell import ZMQInteractiveShell
from IPython import get_ipython
from IPython.display import display
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from tqdm.notebook import tqdm as tqdm_notebook

__all__ = [
    'is_notebook',
    'is_widgets_enabled',
    'is_headless',
    'create_progress_bar',
    'is_colab_env',
    'is_kaggle_env',
    'is_widgets_use_possible',
    'is_terminal_interactive_shell',
    'is_zmq_interactive_shell',
    'ProgressBarGroup'
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
def is_terminal_interactive_shell() -> bool:
    """Check whether we are in a terminal interactive shell or not."""
    return isinstance(get_ipython(), TerminalInteractiveShell)


@lru_cache(maxsize=None)
def is_zmq_interactive_shell() -> bool:
    """Check whether we are in a web-based interactive shell or not."""
    return isinstance(get_ipython(), ZMQInteractiveShell)


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


class PlainNotebookProgressBar(tqdm.tqdm):
    """Custom progress bar."""

    def __init__(self, **kwargs):
        self.display_handler = display({'text/plain': ''}, raw=True, display_id=True)
        kwargs['file'] = io.StringIO()
        super().__init__(**kwargs)

    def refresh(self, nolock=False, lock_args=None):
        """Refresh progress bar."""
        value = super().refresh(nolock, lock_args)
        self.display_handler.update({'text/plain': self.fp.getvalue()}, raw=True)
        self.fp.seek(0)
        return value

    def close(self, *args, **kwargs):
        """Close progress bar."""
        value = super().close(*args, **kwargs)
        self.display_handler.update({'text/plain': ''}, raw=True)
        self.fp.seek(0)
        return value


def create_progress_bar(
    name: str,
    unit: str,
    total: t.Optional[int] = None,
    iterable: t.Optional[t.Sequence[t.Any]] = None,
) -> t.Union[
    tqdm_notebook,
    PlainNotebookProgressBar,
    tqdm.tqdm
]:
    """Create a tqdm progress bar instance."""
    kwargs = {
        'iterable': iterable,
        'total': total,
        'desc': name,
        'unit': f' {unit}',
        'leave': False,
    }

    if iterable is not None:
        iterlen = len(iterable)
    elif total is not None:
        iterlen = total
    else:
        raise ValueError(
            'at least one of the parameters iterable | total must be not None'
        )

    barlen = iterlen if iterlen > 5 else 5

    if is_zmq_interactive_shell() and is_widgets_enabled():
        return tqdm_notebook(
            **kwargs,
            colour='#9d60fb',
            file=sys.stdout
        )

    elif is_zmq_interactive_shell():
        return PlainNotebookProgressBar(
            **kwargs,
            bar_format='{{desc}}:\n|{{bar:{0}}}{{r_bar}}'.format(barlen),  # pylint: disable=consider-using-f-string
        )

    else:
        return tqdm.tqdm(
            **kwargs,
            bar_format='{{desc}}:\n|{{bar:{0}}}{{r_bar}}'.format(barlen),  # pylint: disable=consider-using-f-string
        )


class DummyProgressBar:
    """Dummy progress bar that has only one step."""

    def __init__(self, name: str, unit: str = '') -> None:
        self.pb = create_progress_bar(
            iterable=list(range(1)),
            name=name,
            unit=unit
        )

    def __enter__(self, *args, **kwargs):
        """Enter context."""
        return self

    def __exit__(self, *args, **kwargs):
        """Exit context."""
        for _ in self.pb:
            pass


class ProgressBarGroup:
    """Progress Bar Factory.

    Utility class that makes sure that all progress bars in the
    group will be closed simultaneously.
    """

    register: t.List[t.Union[
        DummyProgressBar,
        tqdm_notebook,
        PlainNotebookProgressBar,
        tqdm.tqdm
    ]]

    def __init__(self) -> None:
        self.register = []

    def create(
        self,
        name: str,
        unit: str,
        total: t.Optional[int] = None,
        iterable: t.Optional[t.Sequence[t.Any]] = None,
    ) -> t.Union[
        tqdm_notebook,
        PlainNotebookProgressBar,
        tqdm.tqdm
    ]:
        """Create progress bar instance."""
        pb = create_progress_bar(
            name=name,
            unit=unit,
            total=total,
            iterable=iterable
        )
        pb.__original_close__, pb.close = (
            pb.close,
            lambda *args, s=pb, **kwargs: s.refresh()
        )
        self.register.append(pb)
        return pb

    def create_dummy(
        self,
        name: str,
        unit: str = ''
    ) -> DummyProgressBar:
        """Create dummy progress bar instance."""
        dpb = DummyProgressBar(name=name, unit=unit)
        dpb.__original_close__, dpb.pb.close = (
            dpb.pb.close,
            lambda *args, s=dpb.pb, **kwargs: s.refresh()
        )
        self.register.append(dpb)
        return dpb

    def __enter__(self, *args, **kwargs):
        """Enter context."""
        return self

    def __exit__(self, *args, **kwargs):
        """Enter context and close all progress bars.."""
        for pb in self.register:
            if hasattr(pb, '__original_close__'):
                pb.__original_close__()


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
