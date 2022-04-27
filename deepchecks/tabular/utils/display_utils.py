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
"""Tabular display utilities."""
import typing as t


__all__ = ['nothing_found_on_columns']


def nothing_found_on_columns(columns: t.List[t.Hashable],
                             message: t.Optional[t.Hashable] = None,
                             max_columns: int = 5):
    """Display columns when nothing is found."""
    display_message = message
    if not display_message:
        display_message = 'Nothing found on columns: '

    for index, col in enumerate(columns):
        if index == max_columns:
            display_message += '...'
            break
        display_message += str(col)
        if index != len(columns) - 1:
            display_message += ', '

    return display_message
