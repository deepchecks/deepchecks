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
"""module contains base logic for the data duplicates checks."""
import typing as t
import abc

from typing_extensions import Self

from deepchecks.core import ConditionCategory, ConditionResult
from deepchecks.utils.strings import format_percent

__all__ = ['DataDuplicatesAbstract']


class DataDuplicatesAbstract(abc.ABC):
    """Base data duplicates check."""

    n_to_show: int
    add_condition: t.Callable[..., t.Any]

    def add_condition_ratio_less_or_equal(self: Self, max_ratio: float = 0) -> Self:
        """Add condition - require duplicate ratio to be less or equal to max_ratio.

        Parameters
        ----------
        max_ratio : float , default: 0
            Maximum ratio of duplicates.
        """
        def max_ratio_condition(result: float) -> ConditionResult:
            details = f'Found {format_percent(result)} duplicate data'
            category = ConditionCategory.PASS if result <= max_ratio else ConditionCategory.WARN
            return ConditionResult(category, details)

        return self.add_condition(f'Duplicate data ratio is less or equal to {format_percent(max_ratio)}',
                                  max_ratio_condition)
