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
"""Alternative way to import tabular checks.

This module exists only for backward compatibility and will be
removed in the next versions.
"""
# flake8: noqa
import warnings

from deepchecks.tabular.checks import *  # pylint: disable=wildcard-import

warnings.warn(
    'Ability to import tabular checks from the `deepchecks.checks` '
    'is deprecated, please import from `deepchecks.tabular.checks` instead',
    DeprecationWarning
)


__all__ = [
    # integrity checks
    'MixedNulls',
    'StringMismatch',
    'MixedDataTypes',
    'IsSingleValue',
    'ClassImbalance',
    'SpecialCharacters',
    'StringLengthOutOfBounds',
    'StringMismatchComparison',
    'DataDuplicates',
    'NewCategoryTrainTest',
    'CategoryMismatchTrainTest',
    'NewLabelTrainTest',
    'ConflictingLabels',
    'OutlierSampleDetection',
    'PercentOfNulls',

    # methodology checks
    'BoostingOverfit',
    'UnusedFeatures',
    'FeatureFeatureCorrelation',
    'FeatureLabelCorrelation',
    'FeatureLabelCorrelationChange',
    'IndexTrainTestLeakage',
    'TrainTestSamplesMix',
    'DateTrainTestLeakageDuplicates',
    'DateTrainTestLeakageOverlap',
    'IdentifierLabelCorrelation',
    'ModelInferenceTime',
    'DatasetsSizeComparison',

    # overview checks
    'ModelInfo',
    'ColumnsInfo',

    # distribution checks
    'TrainTestFeatureDrift',
    'FeatureDrift',
    'TrainTestLabelDrift',
    'LabelDrift',
    'WholeDatasetDrift',
    'TrainTestPredictionDrift',
    'PredictionDrift',
    'MultivariateDrift',

    # performance checks
    'TrainTestPerformance',
    'ConfusionMatrixReport',
    'RocReport',
    'SimpleModelComparison',
    'CalibrationScore',
    'SegmentPerformance',
    'RegressionSystematicError',
    'RegressionErrorDistribution',
    'MultiModelPerformanceReport',
    'WeakSegmentsPerformance',
    'SingleDatasetPerformance'
]
