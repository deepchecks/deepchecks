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
"""Module containing the model evaluation checks in the vision package."""
from .class_performance import ClassPerformance
from .confusion_matrix import ConfusionMatrixReport
from .mean_average_precision_report import MeanAveragePrecisionReport
from .mean_average_recall_report import MeanAverageRecallReport
from .simple_model_comparison import SimpleModelComparison
from .single_dataset_performance import SingleDatasetPerformance
from .train_test_prediction_drift import TrainTestPredictionDrift
from .weak_segments_performance import WeakSegmentsPerformance

__all__ = [
    "TrainTestPredictionDrift",
    "ClassPerformance",
    "ConfusionMatrixReport",
    "MeanAveragePrecisionReport",
    "MeanAverageRecallReport",
    "SimpleModelComparison",
    "SingleDatasetPerformance",
    "WeakSegmentsPerformance"
]
