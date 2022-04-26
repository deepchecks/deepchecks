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
"""Utils module containing useful global functions."""
import re
import subprocess
from functools import lru_cache

import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from IPython import get_ipython  # TODO: I think we should remove ipython from mandatory dependencies


__all__ = ['is_notebook', 'is_widgets_enabled', 'is_headless']


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


class ProgressBar:
    """Progress bar for display while running suite.
    Parameters
    ----------
    name
    length
    """

    def __init__(self, name, length, unit):
        """Initialize progress bar."""
        self.unit = unit
        shared_args = {'total': length, 'desc': name, 'unit': f' {unit}', 'leave': False, 'file': sys.stdout}
        if is_widgets_enabled():
            self.pbar = tqdm_notebook(**shared_args, colour='#9d60fb')
        else:
            # Normal tqdm with colour in notebooks produce bug that the cleanup doesn't remove all characters. so
            # until bug fixed, doesn't add the colour to regular tqdm
            self.pbar = tqdm.tqdm(**shared_args, bar_format=f'{{l_bar}}{{bar:{length}}}{{r_bar}}')

    def set_text(self, text):
        """Set current running check.
        Parameters
        ----------
        text
        """
        self.pbar.set_postfix({self.unit: text})

    def close(self):
        """Close the progress bar."""
        self.pbar.close()

    def inc_progress(self):
        """Increase progress bar value by 1."""
        self.pbar.update(1)
