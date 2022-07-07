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
import logging
import os
import time
import typing as t
from functools import lru_cache

import tqdm
from ipykernel.zmqshell import ZMQInteractiveShell
from IPython import get_ipython
from IPython.display import display
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from tqdm.notebook import tqdm as tqdm_notebook

from deepchecks.utils.logger import get_verbosity

__all__ = [
    'is_notebook',
    'is_headless',
    'create_progress_bar',
    'is_colab_env',
    'is_kaggle_env',
    'is_databricks_env',
    'is_sagemaker_env',
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
def is_colab_env() -> bool:
    """Check if we are in the google colab environment."""
    return 'google.colab' in str(get_ipython())


@lru_cache(maxsize=None)
def is_kaggle_env() -> bool:
    """Check if we are in the kaggle environment."""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None


@lru_cache(maxsize=None)
def is_databricks_env() -> bool:
    """Check if we are in the databricks environment."""
    return 'DATABRICKS_RUNTIME_VERSION' in os.environ


@lru_cache(maxsize=None)
def is_sagemaker_env() -> bool:
    """Check if we are in the AWS Sagemaker environment."""
    return 'AWS_PATH' in os.environ


class HtmlProgressBar:
    """Progress bar implementation that uses html <progress> tag."""

    STYLE = """
    <style>
        progress {
            -webkit-appearance: none;
            border: none;
            border-radius: 3px;
            width: 300px;
            height: 20px;
            vertical-align: middle;
            margin-right: 10px;
            background-color: aliceblue;
        }
        progress::-webkit-progress-bar {
            border-radius: 3px;
            background-color: aliceblue;
        }
        progress::-webkit-progress-value {
            background-color: #9d60fb;
        }
        progress::-moz-progress-bar {
            background-color: #9d60fb;
        }
    </style>
    """

    def __init__(
        self,
        title: str,
        unit: str,
        iterable: t.Iterable[t.Any],
        total: int,
        metadata: t.Optional[t.Mapping[str, t.Any]] = None,
        display_immediately: bool = False,
        disable: bool = False,
    ):
        self._title = title
        self._unit = unit
        self._iterable = iterable
        self._total = total
        self._seconds_passed = 0
        self._inital_metadata = dict(metadata) if metadata else {}
        self._metadata = self._inital_metadata.copy()
        self._progress_bar = None
        self._current_item_index = 0
        display({'text/html': self.STYLE}, raw=True)
        self._display_handler = display({'text/html': ''}, raw=True, display_id=True)
        self._disable = disable
        self._reuse_counter = 0

        if disable is False and display_immediately is True:
            self.refresh()

    def __iter__(self):
        """Iterate over iterable."""
        if self._disable is True:
            try:
                for it in self._iterable:
                    yield it
            finally:
                self._reuse_counter += 1
            return

        if self._reuse_counter > 0:
            self._seconds_passed = 0
            self._current_item_index = 0
            self._progress_bar = None
            self._metadata = self._inital_metadata
            self.clean()

        started_at = time.time()

        try:
            self.refresh()
            for i, it in enumerate(self._iterable, start=1):
                yield it
                self._current_item_index = i
                self._seconds_passed = int(time.time() - started_at)
                self.refresh()
        finally:
            self._reuse_counter += 1
            self.close()

    def refresh(self):
        """Refresh progress bar."""
        self.progress_bar = self.create_progress_bar(
            title=self._title,
            item=self._current_item_index,
            total=self._total,
            seconds_passed=self._seconds_passed,
            metadata=self._metadata
        )
        self._display_handler.update(
            {'text/html': self.progress_bar},
            raw=True
        )

    def close(self):
        """Close progress bar."""
        self._display_handler.update({'text/html': ''}, raw=True)

    def clean(self):
        """Clean display cell."""
        self._display_handler.update({'text/html': ''}, raw=True)

    def set_postfix(self, data: t.Mapping[str, t.Any], refresh: bool = True):
        """Set postfix."""
        self.update_metadata(data, refresh)

    def reset_metadata(self, data: t.Mapping[str, t.Any], refresh: bool = True):
        """Reset metadata."""
        self._metadata = dict(data)
        if refresh is True:
            self.refresh()

    def update_metadata(self, data: t.Mapping[str, t.Any], refresh: bool = True):
        """Update metadata."""
        self._metadata.update(data)
        if refresh is True:
            self.refresh()

    @classmethod
    def create_label(
        cls,
        item: int,
        total: int,
        seconds_passed: int,
        metadata: t.Optional[t.Mapping[str, t.Any]] = None
    ):
        """Create progress bar label."""
        minutes = seconds_passed // 60
        seconds = seconds_passed - (minutes * 60)
        minutes = f'0{minutes}' if minutes < 10 else str(minutes)
        seconds = f'0{seconds}' if seconds < 10 else str(seconds)

        if metadata:
            metadata_string = ', '.join(f'{k}={str(v)}' for k, v in metadata.items())
            metadata_string = f', {metadata_string}'
        else:
            metadata_string = ''

        return f'{item}/{total} [Time: {minutes}:{seconds}{metadata_string}]'

    @classmethod
    def create_progress_bar(
        cls,
        title: str,
        item: int,
        total: int,
        seconds_passed: int,
        metadata: t.Optional[t.Mapping[str, t.Any]] = None
    ) -> str:
        """Create progress bar."""
        return f"""
            <div>
                <label>
                    {title}:<br/>
                    <progress
                        value='{item}'
                        max='{total}'
                        class='deepchecks'
                    >
                    </progress>
                </label>
                <span>{cls.create_label(item, total, seconds_passed, metadata)}</span>
            </div>
        """


def create_progress_bar(
    name: str,
    unit: str,
    total: t.Optional[int] = None,
    iterable: t.Optional[t.Sequence[t.Any]] = None,
) -> t.Union[
    tqdm_notebook,
    HtmlProgressBar,
    tqdm.tqdm
]:
    """Create a progress bar instance."""
    if iterable is not None:
        iterlen = len(iterable)
    elif total is not None:
        iterlen = total
    else:
        raise ValueError(
            'at least one of the parameters iterable | total must be not None'
        )

    is_disabled = get_verbosity() >= logging.WARNING

    if is_zmq_interactive_shell():
        return HtmlProgressBar(
            title=name,
            unit=unit,
            total=iterlen,
            iterable=iterable or range(iterlen),
            display_immediately=True,
            disable=is_disabled
        )
    else:
        barlen = iterlen if iterlen > 5 else 5
        rbar = ' {n_fmt}/{total_fmt} [Time: {elapsed}{postfix}]'
        bar_format = f'{{desc}}:\n|{{bar:{barlen}}}|{rbar}'
        return tqdm.tqdm(
            iterable=iterable,
            total=total,
            desc=name,
            unit=f' {unit}',
            leave=False,
            bar_format=bar_format,
            disable=is_disabled,
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
        HtmlProgressBar,
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
        HtmlProgressBar,
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
