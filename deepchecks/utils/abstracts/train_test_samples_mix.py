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
"""module contains base logic for the train-test samples mix checks."""
import abc
import typing as t

from typing_extensions import Self

from deepchecks.core import ConditionCategory, ConditionResult
from deepchecks.utils.strings import format_percent

__all__ = ['TrainTestSamplesMixAbstract']


class TrainTestSamplesMixAbstract(abc.ABC):
    """Base data duplicates check."""

    add_condition: t.Callable[..., t.Any]

    def add_condition_duplicates_ratio_less_or_equal(self: Self, max_ratio: float = 0.05) -> Self:
        """Add condition - require ratio of test data samples that appear in train data to be less or equal to the\
         threshold.

        Parameters
        ----------
        max_ratio : float , default: 0.05
            Max allowed ratio of test data samples to appear in train data
        """
        def condition(result: dict) -> ConditionResult:
            ratio = result['ratio']
            details = (
                f'Percent of test data samples that appear in train data: {format_percent(ratio)}'
                if ratio
                else 'No samples mix found'
            )
            category = ConditionCategory.PASS if ratio <= max_ratio else ConditionCategory.FAIL
            return ConditionResult(category, details)

        return self.add_condition(
            f'Percentage of test data samples that appear in train data '
            f'is less or equal to {format_percent(max_ratio)}',
            condition
        )
