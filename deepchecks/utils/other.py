# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""module for different utility functions/types."""
import typing as t

__all__ = ["to_ordional_enumeration"]


T = t.TypeVar("T")


def to_ordional_enumeration(data: t.List[T]) -> t.Dict[T, int]:
    """Enumarate each unique item."""
    counter = 0
    enum = {}
    for it in data:
        if it not in enum:
            enum[it] = counter
            counter += 1
    return enum
