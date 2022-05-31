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

__all__ = ['get_max_entry_from_dict', 'sort_dict']


def get_max_entry_from_dict(x: dict):
    """Get from dictionary the entry with maximal value.

    Returns
    -------
    Tuple: key, value
    """
    if not x:
        return None, None
    max_key = max(x, key=x.get)
    return max_key, x[max_key]


def sort_dict(x: dict, reverse=True):
    """Sort dictionary by values.

    Returns
    -------
    Dict: sorted dictionary
    """
    return dict(sorted(x.items(), key=lambda item: item[1], reverse=reverse))
