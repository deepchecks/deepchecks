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
from .performance import ClassPerformance, MeanAveragePrecisionReport, MeanAverageRecallReport, ModelErrorAnalysis, \
                         RobustnessReport, ConfusionMatrixReport, SimpleModelComparison, ImageSegmentPerformance
from .distribution import TrainTestLabelDrift, ImageDatasetDrift, ImagePropertyDrift, TrainTestPredictionDrift, \
    ImagePropertyOutliers, LabelPropertyOutliers, HeatmapComparison
from .methodology import SimpleFeatureContribution, SimilarImageLeakage

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
    'SimilarImageLeakage'
]
