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
"""Module containing the reduce classes and methods."""
import abc
from typing import Dict, Optional

import numpy as np
import pandas as pd

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.logger import get_logger

__all__ = [
    'ReduceMixin',
    'ReduceFeatureMixin',
    'ReducePropertyMixin',
    'ReduceMetricClassMixin'
]


class ReduceMixin(abc.ABC):
    """Mixin for reduce_output function."""

    def reduce_output(self, check_result) -> Dict[str, float]:
        """Return the check result as a reduced dict. Being Used for monitoring.

        Parameters
        ----------
        check_result : CheckResult
            The check result.

        Returns
        -------
        Dict[str, float]
            reduced dictionary in format {str: float} (i.e {'AUC': 0.1}), based on the check's original returned value
        """
        raise NotImplementedError('Must implement reduce_output function')


class ReduceMetricClassMixin(ReduceMixin):
    """Extend ReduceMixin to for performance checks."""


class ReduceFeatureMixin(ReduceMixin):
    """Extend ReduceMixin to identify checks that output result per feature.

    Should implement the feature_reduce function and all the aggregation methods it supports.
    """

    @staticmethod
    def feature_reduce(aggregation_method: str, value_per_feature: pd.Series, feature_importance: Optional[np.array],
                       score_name: str) -> Dict[str, float]:
        """Return an aggregated drift score based on aggregation method defined.

        include the following aggregation options:
        'l2_weighted': L2 norm over the combination of drift scores and feature importance, minus the
        L2 norm of feature importance alone, specifically, ||FI + DRIFT|| - ||FI||. This method returns a
        value between 0 and sqrt(n_features).
        'weighted': Weighted mean based on feature importance, provides a robust estimation on how
        much the drift will affect the model's performance.
        'mean': Mean of all drift scores.
        'max': Maximum of all the features drift scores.
        'none': No averaging. Return a dict with a drift score for each feature.
        'top_5' No averaging. Return a dict with a drift score for top 5 features based on feature importance.
        """
        if aggregation_method == 'none':
            return dict(value_per_feature)
        elif aggregation_method == 'mean':
            return {str('Mean ' + score_name): np.mean(value_per_feature)}
        elif aggregation_method == 'max':
            return {str('Max ' + score_name): np.max(value_per_feature)}

        if aggregation_method in ['weighted', 'l2_weighted', 'top_5'] and feature_importance is None:
            get_logger().warning(
                'Failed to calculate feature importance to all features, using uniform mean instead.')
            return {str('Mean ' + score_name): np.mean(value_per_feature)}
        elif aggregation_method == 'top_5':
            if len(value_per_feature) <= 5:
                return dict(value_per_feature)
            top_5_important = np.flip(np.argsort(feature_importance)[-5:])
            return dict(value_per_feature[top_5_important])
        elif aggregation_method == 'weighted':
            return {str('Weighted ' + score_name): np.sum(np.array(value_per_feature) * feature_importance)}
        elif aggregation_method == 'l2_weighted':
            sum_drift_fi = np.array(value_per_feature) + feature_importance
            return {str('L2 Weighted ' + score_name): np.linalg.norm(sum_drift_fi) - np.linalg.norm(feature_importance)}
        else:
            raise DeepchecksValueError(f'Unknown aggregation method: {aggregation_method}')


class ReducePropertyMixin(ReduceMixin):
    """Extend ReduceMixin to identify checks that output result per property.

    Should implement the property_reduce function and all the aggregation methods it supports.
    """

    @staticmethod
    def property_reduce(aggregation_method: str, value_per_property: pd.Series, score_name: str) -> Dict[str, float]:
        """Return an aggregated drift score based on aggregation method defined.

        include the following aggregation options:
        'mean': Mean of all drift scores.
        'none': No averaging. Return a dict with a drift score for each feature.
        'max': Maximum of all the features drift scores.
        """
        if aggregation_method == 'none':
            return dict(value_per_property)
        elif aggregation_method == 'mean':
            return {str('Mean ' + score_name): np.mean(value_per_property)}
        elif aggregation_method == 'max':
            return {str('Max ' + score_name): np.max(value_per_property)}
        else:
            raise DeepchecksValueError(f'Unknown aggregation method: {aggregation_method}')
