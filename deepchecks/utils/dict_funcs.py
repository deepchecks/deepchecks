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
"""Contain common functions on dictionaries."""

__all__ = ['get_dict_entry_by_value', 'sort_dict']


def get_dict_entry_by_value(x: dict, value_select_fn=max):
    """Get from dictionary the entry with value that returned from value_select_fn.

    Returns
    -------
    Tuple: key, value
    """
    if not x:
        return None, None
    value = value_select_fn(x.values())
    index = list(x.values()).index(value)
    return list(x.keys())[index], value


def sort_dict(x: dict, reverse=True):
    """Sort dictionary by values.

    Returns
    -------
    Dict: sorted dictionary
    """
    return dict(sorted(x.items(), key=lambda item: item[1], reverse=reverse))
