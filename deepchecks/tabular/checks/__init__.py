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
from .data_integrity import (ColumnsInfo, ConflictingLabels, DataDuplicates, FeatureFeatureCorrelation,
                             FeatureLabelCorrelation, IsSingleValue, MixedDataTypes, MixedNulls, OutlierSampleDetection,
                             SpecialCharacters, StringLengthOutOfBounds, StringMismatch)
from .model_evaluation import (BoostingOverfit, CalibrationScore, ConfusionMatrixReport, ModelErrorAnalysis,
                               ModelInferenceTime, ModelInfo, MultiModelPerformanceReport, PerformanceReport,
                               RegressionErrorDistribution, RegressionSystematicError, RocReport, SegmentPerformance,
                               SimpleModelComparison, SingleDatasetPerformance, TrainTestPerformance,
                               TrainTestPredictionDrift, UnusedFeatures, WeakSegmentsPerformance)
from .train_test_validation import (CategoryMismatchTrainTest, DatasetsSizeComparison, DateTrainTestLeakageDuplicates,
                                    DateTrainTestLeakageOverlap, DominantFrequencyChange, FeatureLabelCorrelationChange,
                                    IdentifierLabelCorrelation, IndexTrainTestLeakage, NewLabelTrainTest,
                                    StringMismatchComparison, TrainTestFeatureDrift, TrainTestLabelDrift,
                                    TrainTestSamplesMix, WholeDatasetDrift)

__all__ = [
    # integrity checks
    'MixedNulls',
    'StringMismatch',
    'MixedDataTypes',
    'IsSingleValue',
    'SpecialCharacters',
    'StringLengthOutOfBounds',
    'StringMismatchComparison',
    'DominantFrequencyChange',
    'DataDuplicates',
    'CategoryMismatchTrainTest',
    'NewLabelTrainTest',
    'ConflictingLabels',
    'OutlierSampleDetection',

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
    'WholeDatasetDrift',
    'TrainTestPredictionDrift',

    # performance checks
    'TrainTestPerformance',
    'PerformanceReport',
    'ConfusionMatrixReport',
    'RocReport',
    'SimpleModelComparison',
    'CalibrationScore',
    'SegmentPerformance',
    'RegressionSystematicError',
    'RegressionErrorDistribution',
    'MultiModelPerformanceReport',
    'WeakSegmentsPerformance',
    'ModelErrorAnalysis',
    'SingleDatasetPerformance'
]
