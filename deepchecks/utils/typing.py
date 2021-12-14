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
"""Type definitions."""
# pylint: disable=invalid-hash-returned,invalid-name
from typing_extensions import Protocol, runtime_checkable


__all__ = ['Hashable']


@runtime_checkable
class Hashable(Protocol):
    """Trait for any hashable type that also defines comparison operators."""

    def __hash__(self) -> int: # noqa: D105
        ...
    def __le__(self, __value) -> bool: # noqa: D105
        ...
    def __lt__(self, __value) -> bool: # noqa: D105
        ...
    def __ge__(self, __value) -> bool: # noqa: D105
        ...
    def __gt__(self, __value) -> bool: # noqa: D105
        ...
    def __eq__(self, __value) -> bool: # noqa: D105
        ...
