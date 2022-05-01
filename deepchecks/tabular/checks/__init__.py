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
from .distribution import (TrainTestFeatureDrift, TrainTestLabelDrift,
                           TrainTestPredictionDrift, WholeDatasetDrift)
from .integrity import (CategoryMismatchTrainTest, DataDuplicates,
                        DominantFrequencyChange, IsSingleValue, LabelAmbiguity,
                        MixedDataTypes, MixedNulls, NewLabelTrainTest,
                        OutlierSampleDetection, SpecialCharacters,
                        StringLengthOutOfBounds, StringMismatch,
                        StringMismatchComparison)
from .methodology import (BoostingOverfit, DatasetsSizeComparison,
                          DateTrainTestLeakageDuplicates,
                          DateTrainTestLeakageOverlap, IdentifierLeakage,
                          IndexTrainTestLeakage, ModelInferenceTime,
                          SingleFeatureContribution,
                          SingleFeatureContributionTrainTest,
                          TrainTestSamplesMix, UnusedFeatures)
from .overview import ColumnsInfo, ModelInfo
from .performance import (CalibrationScore, ConfusionMatrixReport,
                          ModelErrorAnalysis, MultiModelPerformanceReport,
                          PerformanceReport, RegressionErrorDistribution,
                          RegressionSystematicError, RocReport,
                          SegmentPerformance, SimpleModelComparison)

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
    'LabelAmbiguity',
    'OutlierSampleDetection',

    # methodology checks
    'BoostingOverfit',
    'UnusedFeatures',
    'SingleFeatureContribution',
    'SingleFeatureContributionTrainTest',
    'IndexTrainTestLeakage',
    'TrainTestSamplesMix',
    'DateTrainTestLeakageDuplicates',
    'DateTrainTestLeakageOverlap',
    'IdentifierLeakage',
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
    'PerformanceReport',
    'ConfusionMatrixReport',
    'RocReport',
    'SimpleModelComparison',
    'CalibrationScore',
    'SegmentPerformance',
    'RegressionSystematicError',
    'RegressionErrorDistribution',
    'MultiModelPerformanceReport',
    'ModelErrorAnalysis'
]
