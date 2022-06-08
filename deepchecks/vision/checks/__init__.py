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
from .data_integrity import ImagePropertyOutliers, LabelPropertyOutliers
from .model_evaluation import (ClassPerformance, ConfusionMatrixReport, ImageSegmentPerformance,
                               MeanAveragePrecisionReport, MeanAverageRecallReport, ModelErrorAnalysis,
                               RobustnessReport, SimpleModelComparison, SingleDatasetScalarPerformance,
                               TrainTestPredictionDrift)
from .train_test_validation import (FeatureLabelCorrelationChange, HeatmapComparison, ImageDatasetDrift,
                                    ImagePropertyDrift, NewLabels, SimilarImageLeakage, TrainTestLabelDrift)

__all__ = ['ClassPerformance', 'ConfusionMatrixReport', 'MeanAveragePrecisionReport', 'MeanAverageRecallReport',
           'RobustnessReport', 'SimpleModelComparison', 'TrainTestLabelDrift', 'ImageDatasetDrift',
           'ImagePropertyDrift', 'ModelErrorAnalysis', 'TrainTestPredictionDrift', 'ImageSegmentPerformance',
           'FeatureLabelCorrelationChange', 'ImagePropertyOutliers', 'LabelPropertyOutliers', 'HeatmapComparison',
           'SimilarImageLeakage', 'NewLabels', 'SingleDatasetScalarPerformance']
