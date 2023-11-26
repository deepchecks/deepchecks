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
"""Module contains checks of model evaluation."""

from deepchecks.tabular.checks.model_evaluation.boosting_overfit import BoostingOverfit
from deepchecks.tabular.checks.model_evaluation.calibration_score import CalibrationScore
from deepchecks.tabular.checks.model_evaluation.confusion_matrix_report import ConfusionMatrixReport
from deepchecks.tabular.checks.model_evaluation.model_inference_time import ModelInferenceTime
from deepchecks.tabular.checks.model_evaluation.model_info import ModelInfo
from deepchecks.tabular.checks.model_evaluation.multi_model_performance_report import MultiModelPerformanceReport
from deepchecks.tabular.checks.model_evaluation.performance_bias import PerformanceBias
from deepchecks.tabular.checks.model_evaluation.prediction_drift import PredictionDrift
from deepchecks.tabular.checks.model_evaluation.regression_error_distribution import RegressionErrorDistribution
from deepchecks.tabular.checks.model_evaluation.regression_systematic_error import RegressionSystematicError
from deepchecks.tabular.checks.model_evaluation.roc_report import RocReport
from deepchecks.tabular.checks.model_evaluation.segment_performance import SegmentPerformance
from deepchecks.tabular.checks.model_evaluation.simple_model_comparison import SimpleModelComparison
from deepchecks.tabular.checks.model_evaluation.single_dataset_performance import SingleDatasetPerformance
from deepchecks.tabular.checks.model_evaluation.train_test_performance import TrainTestPerformance
from deepchecks.tabular.checks.model_evaluation.train_test_prediction_drift import TrainTestPredictionDrift
from deepchecks.tabular.checks.model_evaluation.unused_features import UnusedFeatures
from deepchecks.tabular.checks.model_evaluation.weak_segments_performance import WeakSegmentsPerformance

__all__ = [
    'BoostingOverfit',
    'CalibrationScore',
    'ConfusionMatrixReport',
    'ModelInferenceTime',
    'ModelInfo',
    'MultiModelPerformanceReport',
    'TrainTestPerformance',
    'RegressionErrorDistribution',
    'RegressionSystematicError',
    'RocReport',
    'SegmentPerformance',
    'SimpleModelComparison',
    'TrainTestPredictionDrift',
    'PredictionDrift',
    'WeakSegmentsPerformance',
    'UnusedFeatures',
    'SingleDatasetPerformance',
    'PerformanceBias'
]
