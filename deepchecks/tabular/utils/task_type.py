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
"""Utils module containing enums for tabular related data."""

from enum import Enum

__all__ = [
    'TaskType'
]


class TaskType(Enum):
    """Enum containing supported task types."""

    REGRESSION = 'regression'
    BINARY = 'binary'
    MULTICLASS = 'multiclass'
