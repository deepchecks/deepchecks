# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module importing all checks."""
from .integrity import (
    MixedNulls,
    StringMismatch,
    MixedDataTypes,
    IsSingleValue,
    SpecialCharacters,
    StringLengthOutOfBounds,
    StringMismatchComparison,
    DominantFrequencyChange,
    DataDuplicates,
    CategoryMismatchTrainTest,
    NewLabelTrainTest,
    LabelAmbiguity
)

from .methodology import (
    TrainTestDifferenceOverfit,
    BoostingOverfit,
    UnusedFeatures,
    SingleFeatureContribution,
    SingleFeatureContributionTrainTest,
    IndexTrainTestLeakage,
    TrainTestSamplesMix,
    DateTrainTestLeakageDuplicates,
    DateTrainTestLeakageOverlap,
    IdentifierLeakage,
    ModelInferenceTimeCheck,
    DatasetsSizeComparison
)

from .overview import (
    ModelInfo,
    ColumnsInfo
)

from .distribution import (
    TrustScoreComparison,
    TrainTestFeatureDrift,
    WholeDatasetDrift
)

from .performance import (
    PerformanceReport,
    ConfusionMatrixReport,
    RocReport,
    SimpleModelComparison,
    CalibrationScore,
    SegmentPerformance,
    RegressionSystematicError,
    RegressionErrorDistribution,
    ClassPerformance,
    MultiModelPerformanceReport
)


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

    # methodology checks
    'TrainTestDifferenceOverfit',
    'BoostingOverfit',
    'UnusedFeatures',
    'SingleFeatureContribution',
    'SingleFeatureContributionTrainTest',
    'IndexTrainTestLeakage',
    'TrainTestSamplesMix',
    'DateTrainTestLeakageDuplicates',
    'DateTrainTestLeakageOverlap',
    'IdentifierLeakage',
    'ModelInferenceTimeCheck',
    'DatasetsSizeComparison',

    # overview checks
    'ModelInfo',
    'ColumnsInfo',

    # distribution checks
    'TrustScoreComparison',
    'TrainTestFeatureDrift',
    'WholeDatasetDrift',

    # performance checks
    'PerformanceReport',
    'ConfusionMatrixReport',
    'RocReport',
    'SimpleModelComparison',
    'CalibrationScore',
    'SegmentPerformance',
    'RegressionSystematicError',
    'RegressionErrorDistribution',
    'ClassPerformance',
    'MultiModelPerformanceReport'
]
