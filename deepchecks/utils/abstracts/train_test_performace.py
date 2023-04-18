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
"""Module containing the train test performance check."""
import typing as t
import abc

from typing_extensions import Self

from deepchecks.core.check_utils.class_performance_utils import (
    get_condition_class_performance_imbalance_ratio_less_than, 
    get_condition_test_performance_greater_than,
    get_condition_train_test_relative_degradation_less_than
)
from deepchecks.utils.strings import format_percent

__all__ = ['TrainTestPerformanceAbstract']


class TrainTestPerformanceAbstract:
    """Base functionality for some train-test performance checks."""

    @classmethod
    @abc.abstractmethod
    def _default_per_class_scorers(cls) -> t.Sequence[str]:
        raise NotImplementedError()

    def add_condition_test_performance_greater_than(self: Self, min_score: float) -> Self:
        """Add condition - metric scores are greater than the threshold.

        Parameters
        ----------
        min_score : float
            Minimum score to pass the check.
        """
        condition = get_condition_test_performance_greater_than(min_score=min_score)
        return self.add_condition(f'Scores are greater than {min_score}', condition)

    def add_condition_train_test_relative_degradation_less_than(self: Self, threshold: float = 0.1) -> Self:
        """Add condition - test performance is not degraded by more than given percentage in train.

        Parameters
        ----------
        threshold : float , default: 0.1
            maximum degradation ratio allowed (value between 0 and 1)
        """
        name = f'Train-Test scores relative degradation is less than {threshold}'
        condition = get_condition_train_test_relative_degradation_less_than(threshold=threshold)
        return self.add_condition(name, condition)

    def add_condition_class_performance_imbalance_ratio_less_than(
        self: Self,
        threshold: float = 0.3,
        score: t.Optional[str] = None
    ) -> Self:
        """Add condition - relative ratio difference between highest-class and lowest-class is less than threshold.

        Parameters
        ----------
        threshold : float , default: 0.3
            ratio difference threshold
        score : str , default: None
            limit score for condition

        Returns
        -------
        Self
            instance of 'TrainTestPerformance' or it subtype

        Raises
        ------
        DeepchecksValueError
            if unknown score function name were passed.
        """
        if score is None:
            score = next(iter(self._default_per_class_scorers()))

        name = f"Relative ratio difference between labels '{score}' score is less than {format_percent(threshold)}"
        condition = get_condition_class_performance_imbalance_ratio_less_than(threshold=threshold, score=score)
        return self.add_condition(name=name, condition_func=condition)
