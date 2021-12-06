# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""Utils module containing useful global functions."""
import re
import subprocess
from functools import lru_cache

from IPython import get_ipython # TODO: I think we should remove ipython from mandatory dependencies


__all__ = ['is_notebook', 'is_widgets_enabled']


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
        except: # pylint: disable=bare-except
            return False
