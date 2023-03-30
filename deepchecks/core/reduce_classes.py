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
# pylint: disable=import-outside-toplevel
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
    'ReduceLabelMixin',
    'ReduceMetricClassMixin'
]


class ReduceMixin(abc.ABC):
    """Mixin for reduce_output function."""

    def greater_is_better(self):
        """Return True if the check reduce_output is better when it is greater."""
        raise NotImplementedError('Must implement greater_is_better function')

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


class ReduceLabelMixin(ReduceMixin):
    """Extend ReduceMixin to for checks that use the labels."""

    pass


class ReduceMetricClassMixin(ReduceLabelMixin):
    """Extend ReduceMixin to for performance checks."""

    def greater_is_better(self):
        """Return True if the check reduce_output is better when it is greater.

        Returns False if the check is a regression check and the metric is in the lower_is_better list, else True.
        """
        from deepchecks.tabular.metric_utils.scorers import regression_scorers_lower_is_better_dict

        lower_is_better_names = set(regression_scorers_lower_is_better_dict.keys())
        if not hasattr(self, 'scorers'):
            raise NotImplementedError('ReduceMetricClassMixin must be used with a check that has a scorers attribute')
        elif isinstance(self.scorers, dict):
            names = list(self.scorers.keys())
        elif isinstance(self.scorers, list):
            names = self.scorers
        else:
            raise NotImplementedError('ReduceMetricClassMixin must be used with a check that has a scorers attribute'
                                      ' of type DeepcheckScorer or dict')

        names = [x.lower().replace(' ', '_') for x in names]
        if all((name in lower_is_better_names) for name in names):
            return False
        elif all((name not in lower_is_better_names) for name in names):
            return True
        else:
            raise DeepchecksValueError('Cannot reduce metric class with mixed scorers')


class ReduceFeatureMixin(ReduceMixin):
    """Extend ReduceMixin to identify checks that output result per feature.

    Should implement the feature_reduce function and all the aggregation methods it supports.
    """

    def greater_is_better(self):
        """Return True if the check reduce_output is better when it is greater."""
        return False

    @staticmethod
    def feature_reduce(aggregation_method: str, value_per_feature: pd.Series, feature_importance: Optional[pd.Series],
                       score_name: str) -> Dict[str, float]:
        """Return an aggregated drift score based on aggregation method defined."""
        if aggregation_method is None or aggregation_method == 'none':
            return dict(value_per_feature)
        elif aggregation_method == 'mean':
            return {str('Mean ' + score_name): np.mean(value_per_feature)}
        elif aggregation_method == 'max':
            return {str('Max ' + score_name): np.max(value_per_feature)}

        if aggregation_method not in ['weighted', 'l3_weighted', 'l5_weighted']:
            raise DeepchecksValueError(f'Unknown aggregation method: {aggregation_method}')
        elif feature_importance is None or feature_importance.isna().values.any():
            get_logger().warning('Failed to calculate feature importance, using uniform mean instead.')
            return {str(str.title(aggregation_method.replace('_', ' ')) + ' ' + score_name): np.mean(value_per_feature)}
        else:
            value_per_feature = value_per_feature[feature_importance.index]
            feature_importance = feature_importance[value_per_feature.notna().values]
            value_per_feature.dropna(inplace=True)
            value_per_feature, feature_importance = np.asarray(value_per_feature), np.asarray(feature_importance)

        if aggregation_method == 'weighted':
            return {str('Weighted ' + score_name): np.sum(value_per_feature * feature_importance)}
        elif aggregation_method == 'l3_weighted':
            return {str('L3 Weighted ' + score_name): np.sum((value_per_feature ** 3) * feature_importance) ** (1. / 3)}
        elif aggregation_method == 'l5_weighted':
            return {str('L5 Weighted ' + score_name): np.sum((value_per_feature ** 5) * feature_importance) ** (1. / 5)}


class ReducePropertyMixin(ReduceMixin):
    """Extend ReduceMixin to identify checks that output result per property.

    Should implement the property_reduce function and all the aggregation methods it supports.
    """

    def greater_is_better(self):
        """Return True if the check reduce_output is better when it is greater."""
        return False

    @staticmethod
    def property_reduce(aggregation_method: str, value_per_property: pd.Series, score_name: str) -> Dict[str, float]:
        """Return an aggregated drift score based on aggregation method defined."""
        if aggregation_method is None or aggregation_method == 'none':
            return dict(value_per_property)
        elif aggregation_method == 'mean':
            return {str('Mean ' + score_name): np.mean(value_per_property)}
        elif aggregation_method == 'max':
            return {str('Max ' + score_name): np.max(value_per_property)}
        else:
            raise DeepchecksValueError(f'Unknown aggregation method: {aggregation_method}')
