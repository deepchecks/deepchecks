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
"""Tabular checks' messages utilities."""
from typing import Sized

__all__ = ['get_condition_passed_message']


def get_condition_passed_message(sized):
    """Get a message for a condition that passed that specifies the number of columns passed."""
    if isinstance(sized, int):
        num_columns = sized
    elif isinstance(sized, Sized):
        num_columns = len(sized)
    else:
        raise TypeError('sized must be an int or a Sized')

    if num_columns == 0:
        return 'No relevant columns to check were found'

    message = f'Passed for {num_columns} relevant column'
    if num_columns > 1:
        message += 's'
    return message
