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
"""
Module contains checks of model performance metrics.

.. deprecated:: 0.7.0
        `deepchecks.tabular.checks.performance is deprecated and will be removed in deepchecks 0.8 version.
        Use `deepchecks.tabular.checks.model_evaluation` instead.
"""
import warnings

from ..model_evaluation import (CalibrationScore, ConfusionMatrixReport, ModelErrorAnalysis,
                                MultiModelPerformanceReport, RegressionErrorDistribution, RegressionSystematicError,
                                RocReport, SegmentPerformance, SimpleModelComparison, SingleDatasetPerformance,
                                TrainTestPerformance)

__all__ = [
    'TrainTestPerformance',
    'ConfusionMatrixReport',
    'RocReport',
    'SimpleModelComparison',
    'CalibrationScore',
    'SegmentPerformance',
    'RegressionSystematicError',
    'RegressionErrorDistribution',
    'MultiModelPerformanceReport',
    'ModelErrorAnalysis',
    'SingleDatasetPerformance'
]

warnings.warn(
                'deepchecks.tabular.checks.performance is deprecated. Use deepchecks.tabular.checks.model_evaluation '
                'instead.',
                DeprecationWarning
            )
