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
"""The base abstract functionality for features drift checks."""
import abc

from typing_extensions import Self

from deepchecks.utils.distribution.drift import drift_condition

__all__ = ["FeatureDriftAbstract"]


class FeatureDriftAbstract(abc.ABC):

    def add_condition_drift_score_less_than(
        self: Self,
        max_allowed_categorical_score: float = 0.2,
        max_allowed_numeric_score: float = 0.2,
        allowed_num_features_exceeding_threshold: int = 0
    ) -> Self:
        """
        Add condition - require drift score to be less than the threshold.

        The industry standard for PSI limit is above 0.2.
        There are no common industry standards for other drift methods, such as Cramer's V,
        Kolmogorov-Smirnov and Earth Mover's Distance.

        Parameters
        ----------
        max_allowed_categorical_score: float , default: 0.2
            The max threshold for the categorical variable drift score
        max_allowed_numeric_score: float ,  default: 0.2
            The max threshold for the numeric variable drift score
        allowed_num_features_exceeding_threshold: int , default: 0
            Determines the number of features with drift score above threshold needed to fail the condition.

        Returns
        -------
        ConditionResult
            False if more than allowed_num_features_exceeding_threshold drift scores are above threshold, True otherwise
        """
        condition = drift_condition(max_allowed_categorical_score, max_allowed_numeric_score, 'column', 'columns',
                                    allowed_num_features_exceeding_threshold)

        return self.add_condition(f'categorical drift score < {max_allowed_categorical_score} and '
                                  f'numerical drift score < {max_allowed_numeric_score}',
                                  condition)
