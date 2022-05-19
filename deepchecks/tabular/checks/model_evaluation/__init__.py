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
"""Module contains checks of model evaluation."""
from .boosting_overfit import BoostingOverfit
from .calibration_score import CalibrationScore
from .confusion_matrix_report import ConfusionMatrixReport
from .model_error_analysis import ModelErrorAnalysis
from .model_inference_time import ModelInferenceTime
from .model_info import ModelInfo
from .multi_model_performance_report import MultiModelPerformanceReport
from .performance_report import PerformanceReport
from .regression_error_distribution import RegressionErrorDistribution
from .regression_systematic_error import RegressionSystematicError
from .roc_report import RocReport
from .segment_performance import SegmentPerformance
from .simple_model_comparison import SimpleModelComparison
from .train_test_prediction_drift import TrainTestPredictionDrift
from .unused_features import UnusedFeatures

__all__ = [
    'BoostingOverfit',
    'CalibrationScore',
    'ConfusionMatrixReport',
    'ModelErrorAnalysis',
    'ModelInferenceTime',
    'ModelInfo',
    'MultiModelPerformanceReport',
    'PerformanceReport',
    'RegressionErrorDistribution',
    'RegressionSystematicError',
    'RocReport',
    'SegmentPerformance',
    'SimpleModelComparison',
    'TrainTestPredictionDrift',
    'UnusedFeatures'
]
