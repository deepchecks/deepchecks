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
"""Module importing all vision checks."""
from .distribution import (HeatmapComparison,NewLabels, ImageDatasetDrift,
                           ImagePropertyDrift, ImagePropertyOutliers,
                           LabelPropertyOutliers, TrainTestLabelDrift,
                           TrainTestPredictionDrift)
from .methodology import SimilarImageLeakage, SimpleFeatureContribution
from .performance import (ClassPerformance, ConfusionMatrixReport,
                          ImageSegmentPerformance, MeanAveragePrecisionReport,
                          MeanAverageRecallReport, ModelErrorAnalysis,
                          RobustnessReport, SimpleModelComparison)

__all__ = [
    'ClassPerformance',
    'ConfusionMatrixReport',
    'MeanAveragePrecisionReport',
    'MeanAverageRecallReport',
    'RobustnessReport',
    'SimpleModelComparison',
    'TrainTestLabelDrift',
    'ImageDatasetDrift',
    'ImagePropertyDrift',
    'ModelErrorAnalysis',
    'TrainTestPredictionDrift',
    'ImageSegmentPerformance',
    'SimpleFeatureContribution',
    'ImagePropertyOutliers',
    'LabelPropertyOutliers',
    'HeatmapComparison',
    'SimilarImageLeakage',
    'NewLabels'
]
