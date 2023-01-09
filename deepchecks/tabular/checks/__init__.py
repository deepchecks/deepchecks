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
"""Module importing all tabular checks."""
from deepchecks.tabular.checks.data_integrity import PercentOfNulls

from .data_integrity import (ClassImbalance, ColumnsInfo, ConflictingLabels, DataDuplicates, FeatureFeatureCorrelation,
                             FeatureLabelCorrelation, IsSingleValue, MixedDataTypes, MixedNulls, OutlierSampleDetection,
                             SpecialCharacters, StringLengthOutOfBounds, StringMismatch)
from .model_evaluation import (BoostingOverfit, CalibrationScore, ConfusionMatrixReport, ModelInferenceTime, ModelInfo,
                               MultiModelPerformanceReport, RegressionErrorDistribution, RegressionSystematicError,
                               RocReport, SegmentPerformance, SimpleModelComparison, SingleDatasetPerformance,
                               TrainTestPerformance, TrainTestPredictionDrift, UnusedFeatures, WeakSegmentsPerformance)
from .train_test_validation import (CategoryMismatchTrainTest, DatasetsSizeComparison, DateTrainTestLeakageDuplicates,
                                    DateTrainTestLeakageOverlap, FeatureLabelCorrelationChange,
                                    IdentifierLabelCorrelation, IndexTrainTestLeakage, MultivariateDrift,
                                    NewCategoryTrainTest, NewLabelTrainTest, StringMismatchComparison,
                                    TrainTestFeatureDrift, TrainTestLabelDrift, TrainTestSamplesMix, WholeDatasetDrift)

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
    'NewLabelTrainTest',
    'ConflictingLabels',
    'OutlierSampleDetection',
    'PercentOfNulls',
    'CategoryMismatchTrainTest',

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
    'TrainTestLabelDrift',
    'MultivariateDrift',
    'WholeDatasetDrift',
    'TrainTestPredictionDrift',

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
