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
import sys
from functools import lru_cache
from importlib import import_module


from IPython import get_ipython  # TODO: I think we should remove ipython from mandatory dependencies


__all__ = ['is_notebook', 'is_ipython_display']


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
