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
"""Type definitions."""
from typing_extensions import Protocol, runtime_checkable


__all__ = ['Hashable']


@runtime_checkable
class Hashable(Protocol):
    """Trait for any hashable type that also defines comparison operators."""

    def __hash__(self) -> int: # pylint: disable=invalid-hash-returned, noqa: D105
        ...
    def __le__(self, value) -> bool: # noqa: D105
        ...
    def __lt__(self, value) -> bool: # noqa: D105
        ...
    def __ge__(self, value) -> bool: # noqa: D105
        ...
    def __gt__(self, value) -> bool: # noqa: D105
        ...
    def __eq__(self, value) -> bool: # noqa: D105
        ...
