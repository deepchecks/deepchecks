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
"""Module importing all vision checks."""
from .data_integrity import ImagePropertyOutliers, LabelPropertyOutliers, PropertyLabelCorrelation
from .model_evaluation import (ClassPerformance, ConfusionMatrixReport, MeanAveragePrecisionReport,
                               MeanAverageRecallReport, PredictionDrift, SimpleModelComparison,
                               SingleDatasetPerformance, TrainTestPredictionDrift, WeakSegmentsPerformance)
from .train_test_validation import (HeatmapComparison, ImageDatasetDrift, ImagePropertyDrift, LabelDrift, NewLabels,
                                    PropertyLabelCorrelationChange, TrainTestLabelDrift)

__all__ = ['ClassPerformance', 'ConfusionMatrixReport', 'MeanAveragePrecisionReport', 'MeanAverageRecallReport',
           'SimpleModelComparison', 'TrainTestLabelDrift', 'LabelDrift', 'ImageDatasetDrift',
           'ImagePropertyDrift', 'TrainTestPredictionDrift', 'PredictionDrift',
           'PropertyLabelCorrelationChange', 'ImagePropertyOutliers', 'LabelPropertyOutliers', 'HeatmapComparison',
           'NewLabels', 'SingleDatasetPerformance', 'PropertyLabelCorrelation', 'WeakSegmentsPerformance']
