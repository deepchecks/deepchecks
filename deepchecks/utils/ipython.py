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
# pylint: disable=assignment-from-none,broad-except,import-outside-toplevel
"""Utils module containing useful global functions."""
import io
import json
import os
import subprocess
import sys
import typing as t
import urllib.request
import warnings
from functools import lru_cache
from urllib.parse import parse_qs, urlparse

import tqdm
from ipykernel.zmqshell import ZMQInteractiveShell
from IPython import get_ipython
from IPython.display import display
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from tqdm.notebook import tqdm as tqdm_notebook
from typing_extensions import TypedDict

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
    warnings.warn('', category=DeprecationWarning)
    return is_widgets_use_possible()


@lru_cache(maxsize=None)
def is_colab_env() -> bool:
    """Check if we are in the google colab enviroment."""
    return 'google.colab' in str(get_ipython())


@lru_cache(maxsize=None)
def is_kaggle_env() -> bool:
    """Check if we are in the kaggle enviroment."""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None


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


class JupyterLabExtensionInfo(TypedDict):
    name: str
    enabled: bool
    installed_version: str
    status: str


class NotebookExtensionInfo(TypedDict):
    name: str
    enabled: bool
    status: str


def is_jupyter_server_extension_enabled(name: str) -> bool:
    """Find out whether provided jupyter server extension is enabled."""
    extensions = get_jupyter_server_extensions()

    if extensions is None:
        return False

    if name not in extensions:
        return False

    extension = extensions.get(name)

    return (
        extension.enabled and extension.validate()
        if extension is not None
        else False
    )


def get_jupyter_server_extensions() -> t.Optional[t.Mapping[str, t.Any]]:
    """Get dictionary of jupyter server extensions.

    Returns
    -------
    None :
        when 'jupyter_server' and 'jupyter_core' are not available,
        it means that 'Jupyter' is not installed
    Mapping[str, jupyter_server.extension.manager.ExtensionPackage] :
        map of extension name -> extension package instance
    """
    try:
        from jupyter_core.paths import jupyter_config_path
        from jupyter_server.extension.config import ExtensionConfigManager
        from jupyter_server.extension.manager import ExtensionManager
    except ImportError:
        return
    else:
        folders = t.cast(t.List[str], jupyter_config_path())
        config_manager = ExtensionConfigManager(read_config_path=folders)
        extension_manager = ExtensionManager(config_manager=config_manager)
        return extension_manager.extensions


def get_jupyter_server_url() -> t.Optional[str]:
    """Get running jupyter server url.

    Returns
    -------
    None :
        when there is no running jupyter server instance,
        or when there is more than one running jupyter
        server instance. In the second case, we cannot
        determine which one is ours.
    str :
        jupyter server url string
    """
    try:
        output = subprocess.getoutput('jupyter server list').split('\n')
    except BaseException:
        return
    else:
        urls = [
            line
            for line in output
            if line.strip().startswith('http') or line.strip().startswith('https')
        ]

        if len(urls) > 1:
            warnings.warn('')  # TODO:
            return
        if len(urls) == 0:
            return

        url = urls[0].split('::')[0].strip()
        return url.split(' ')[0]


def extract_jupyter_server_token(url: str) -> str:
    """Extract token string from jupyter server url query params string."""
    query = parse_qs(url)
    token = (query.get('token') or [])
    return (token[0] if len(token) > 0 else '')


def is_jupyterlab_extension_enabled(name: str) -> bool:
    """Find out whether provided nbclassic extension is enabled."""
    server_url = get_jupyter_server_url()
    extensions = None

    if server_url is not None:
        extensions = request_jupyterlab_extensions(server_url)

    if extensions is None:
        extensions = get_jupyterlab_extensions()

    return (
        extensions[name]['enabled'] is True and extensions[name]['status'].upper() == 'OK'
        if extensions is not None and name in extensions
        else False
    )


def request_jupyterlab_extensions(server_url: str) -> t.Optional[t.Mapping[str, JupyterLabExtensionInfo]]:
    """Request dictionary of jupyterlab extensions from the jupyter server.

    Parameters
    ----------
    server_url : str
        jupyter server url

    Returns
    -------
    None :
        if an error is raised during output parsing or cmd execution
    Mapping[str, JupyterLabExtensionInfo] :
        map of extension name -> extension info
    """
    urlobj = urlparse(server_url)
    url = '{}://{}/lab/api/extensions?token={}'.format(  # pylint: disable=consider-using-f-string
        urlobj.scheme,
        urlobj.netloc,
        extract_jupyter_server_token(urlobj.query)
    )
    try:
        with urllib.request.urlopen(url) as f:
            return {e['name']: e for e in json.load(f)}
    except BaseException:
        return


def get_jupyterlab_extensions(merge: bool = True) -> t.Optional[t.Mapping[str, t.Any]]:
    """Get list of jupyterlab extensions by executing extension manager cli command.

    Parameters
    ----------
    merge : bool, default True
        whether to merge configurations from different directories or not.
        Jupyter uses several locations for configuration storing and each
        of them has a different priority, so if the same configuration option
        is met in different configuration files, then will be used option
        from the file with the higher priority. If set to False will return
        list of extensions settings from each configuration storage directory.

    Returns
    -------
    None :
        if an error is raised during output parsing or cmd execution
    Mapping[str, List[JupyterLabExtensionInfo]] :
        map of configuration files -> list of extension if 'merge' is set to False
    Mapping[str, JupyterLabExtensionInfo] :
        map of extension name -> extension info if 'merge' is set to True

    Output of the cmd has next format:

        > JupyterLab v3.4.2
        > /home/user/.local/share/jupyter/labextensions
        >   jupyterlab-plotly v5.5.0 enabled OK
        >   @jupyter-widgets/jupyterlab-manager v3.0.1 disabled OK (python, jupyterlab_widgets)
        >
        > /home/user/Projects/deepchecks/venv/share/jupyter/labextensions
        >   jupyterlab_pygments v0.2.2 enabled OK (python, jupyterlab_pygments)
        >   catboost-widget v1.0.0 enabled OK
    """
    try:
        output = subprocess.getoutput('jupyter labextension list').split('\n')
    except BaseException:
        return
    else:
        data = {}

        try:
            line_index = 0
            output_len = len(output)
            while line_index < output_len:
                # look for a line with path to the config directory
                line = output[line_index]
                is_config_directory_line = '/labextensions' in line
                if not is_config_directory_line:
                    # unknown line - skip
                    line_index += 1
                    continue
                else:
                    # collect extensions that are printed below config directory line
                    config_folder = line.strip()
                    extensions = []
                    line_index += 1
                    while line_index < output_len:
                        line = output[line_index]
                        is_extension_line = 'enabled' in line or 'disabled' in line
                        if not is_extension_line:
                            # unknown line, no more info about extensions
                            # go back to the outer loop
                            break
                        else:
                            # parse extension info line
                            line_index += 1
                            name, version, enabled, status, *_ = line.strip().split(' ')
                            extensions.append(JupyterLabExtensionInfo(
                                name=name,
                                installed_version=version,
                                enabled='enabled' in enabled,
                                status='OK' if 'OK' in status else ''
                            ))
                    data[config_folder] = extensions
        except ValueError:
            return

        if not merge:
            return data

        return dict(
            (extension['name'], {'folder': folder_name, **extension})
            for folder_name, extensions in list(data.items())[::-1]
            for extension in extensions
        )


def is_nbclassic_extension_enabled(name: str) -> bool:
    """Find out whether provided nbclassic extension is enabled."""
    server_url = get_jupyter_server_url()
    extensions = None

    if server_url is not None:
        extensions = request_nbclassic_extensions(server_url)

    if extensions is None:
        extensions = get_nbclassic_extensions()

    return (
        extensions[name]['enabled'] is True and extensions[name]['status'].upper() == 'OK'
        if extensions is not None and name in extensions
        else False
    )


def request_nbclassic_extensions(server_url: str) -> t.Optional[t.Mapping[str, NotebookExtensionInfo]]:
    """Request a dictionary of nbclassic extensions from the jupyter server.

    Parameters
    ----------
    server_url : str
        jupyter server url

    Returns
    -------
    None :
        if an error is raised during execution.
    Mapping[str, JupyterLabExtensionInfo] :
        map of extension name -> extension info
    """
    urlobj = urlparse(server_url)
    url = '{}://{}/api/config/notebook?token={}'.format(  # pylint: disable=consider-using-f-string
        urlobj.scheme,
        urlobj.netloc,
        extract_jupyter_server_token(urlobj.query)
    )
    try:
        with urllib.request.urlopen(url) as f:
            data = json.load(f)
            output = {}
            if not data:
                return {}
            for k, v in data['load_extensions'].items():
                name = k.replace('/extension', '')
                output[name] = NotebookExtensionInfo(name=name, enabled=v, status='OK')
            return output
    except BaseException:
        return


def get_nbclassic_extensions(merge: bool = True) -> t.Optional[t.Mapping[str, t.Any]]:
    """Get list of nbclassic extensions.

    Parameters
    ----------
    merge : bool, default True
        whether to merge configurations from different directories or not.
        Jupyter uses several locations for configuration storing and each
        of them has a different priority, so if the same configuration option
        is met in different configuration files, then will be used option
        from the file with the higher priority. If set to False will return
        list of extensions settings from each configuration storage directory.

    Returns
    -------
    None :
        if an error is raised during execution
    Mapping[str, List[NotebookExtensionInfo]] :
        map of configuration files -> list of extension if 'merge' is set to False
    Mapping[str, NotebookExtensionInfo] :
        map of extension name -> extension info if 'merge' is set to True
    """
    try:
        from jupyter_core.paths import jupyter_config_path
        from notebook.config_manager import BaseJSONConfigManager
        from notebook.nbextensions import validate_nbextension
    except ImportError:
        return

    directories = [os.path.join(p, 'nbconfig') for p in jupyter_config_path()]
    data = {}

    try:
        for d in directories:
            config_manager = BaseJSONConfigManager(config_dir=d)
            config = t.cast(t.Optional[t.Dict[str, t.Any]], config_manager.get('notebook'))

            if config:
                extensions = config.get('load_extensions')
                if extensions:
                    data[d] = [
                        NotebookExtensionInfo(
                            name=name,
                            enabled=is_enabled,
                            status='OK' if len(validate_nbextension(name)) == 0 else ''
                        )
                        for name, is_enabled in extensions.items()
                    ]
    except BaseException:
        return

    if not merge:
        return data

    return dict(
        (extension['name'], {'folder': folder_name, **extension})
        for folder_name, extensions in list(data.items())[::-1]
        for extension in extensions
    )


def is_widgets_use_possible() -> bool:
    """Find out whether ipywidgets use is possible within jupyter interactive REPL."""
    is_jupyterlab_enabled = is_jupyter_server_extension_enabled('jupyterlab')
    is_nbclassic_enabled = is_jupyter_server_extension_enabled('nbclassic')

    if is_jupyterlab_enabled and is_nbclassic_enabled:
        condition = (
            is_jupyterlab_extension_enabled('@jupyter-widgets/jupyterlab-manager'),
            is_nbclassic_extension_enabled('jupyter-js-widgets')
        )
        if all(condition):
            return True
        elif any(condition):
            warnings.warn('')  # TODO:
            return True
        else:
            return False
    elif is_jupyterlab_enabled:
        return is_jupyterlab_extension_enabled('@jupyter-widgets/jupyterlab-manager')
    elif is_nbclassic_enabled:
        return is_nbclassic_extension_enabled('jupyter-js-widgets')
    else:
        return False


def is_interactive_output_use_possible() -> bool:
    """Find out whether the use of interactive outputs is possible within jupyter interactive REPL."""
    is_jupyterlab_enabled = is_jupyter_server_extension_enabled('jupyterlab')
    is_nbclassic_enabled = is_jupyter_server_extension_enabled('nbclassic')

    if is_jupyterlab_enabled and is_nbclassic_enabled:
        is_jupyterlab_requirements_met = (
            is_jupyterlab_extension_enabled('@jupyter-widgets/jupyterlab-manager')
            and is_jupyterlab_extension_enabled('jupyterlab-plotly')
        )
        condition = (
            is_jupyterlab_requirements_met,
            is_nbclassic_extension_enabled('jupyter-js-widgets')
        )
        if all(condition):
            return True
        elif any(condition):
            warnings.warn('')  # TODO:
            return True
        else:
            return False
    elif is_jupyterlab_enabled:
        return (
            is_jupyterlab_extension_enabled('@jupyter-widgets/jupyterlab-manager')
            and is_jupyterlab_extension_enabled('jupyterlab-plotly')
        )
    elif is_nbclassic_enabled:
        return is_nbclassic_extension_enabled('jupyter-js-widgets')
    else:
        return False
