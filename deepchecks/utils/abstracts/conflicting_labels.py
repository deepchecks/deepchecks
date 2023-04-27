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
"""Module contains an common functionality for conflicting labels checks."""
import typing as t

from typing_extensions import Self

from deepchecks.core import ConditionCategory, ConditionResult
from deepchecks.utils.strings import format_percent

__all__ = ['ConflictingLabelsAbstract']


class ConflictingLabelsAbstract:
    """Base functionality for conflicting labels checks."""

    add_condition: t.Callable[..., t.Any]

    def add_condition_ratio_of_conflicting_labels_less_or_equal(self: Self, max_ratio=0) -> Self:
        """Add condition - require ratio of samples with conflicting labels less or equal to max_ratio.

        Parameters
        ----------
        max_ratio : float , default: 0
            Maximum ratio of samples with multiple labels.
        """
        def max_ratio_condition(result: t.Dict[str, t.Any]) -> ConditionResult:
            percent = t.cast(float, result['percent_of_conflicting_samples'])
            details = f'Ratio of samples with conflicting labels: {format_percent(percent)}'
            category = ConditionCategory.PASS if percent <= max_ratio else ConditionCategory.FAIL
            return ConditionResult(category, details)

        return self.add_condition(
            f'Ambiguous sample ratio is less or equal to {format_percent(max_ratio)}',
            max_ratio_condition
        )
