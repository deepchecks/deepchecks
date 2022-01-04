# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
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
import sys
from functools import lru_cache
from importlib import import_module

from IPython import get_ipython  # TODO: I think we should remove ipython from mandatory dependencies


__all__ = ['is_notebook', 'is_ipython_display', 'is_widgets_enabled']


@lru_cache(maxsize=None)
def is_notebook() -> bool:
    """Check if we're in an interactive context (Notebook, GUI support) or terminal-based.

    Returns:
        True if we are in a notebook context, False otherwise
    """
    try:
        shell = get_ipython()
        return hasattr(shell, 'config')
    except NameError:
        return False  # Probably standard Python interpreter


@lru_cache(maxsize=None)
def is_ipython_display() -> bool:
    """Check whether we have IPython display module in current environment."""
    module = 'IPython.display'
    if module in sys.modules:
        return True
    try:
        import_module(module)
        return True
    # pylint: disable=broad-except
    except Exception:
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
