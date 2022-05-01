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
"""Module contains checks of model performance metrics."""
from .calibration_score import CalibrationScore
from .confusion_matrix_report import ConfusionMatrixReport
from .model_error_analysis import ModelErrorAnalysis
from .multi_model_performance_report import MultiModelPerformanceReport
from .performance_report import PerformanceReport
from .regression_error_distribution import RegressionErrorDistribution
from .regression_systematic_error import RegressionSystematicError
from .roc_report import RocReport
from .segment_performance import SegmentPerformance
from .simple_model_comparison import SimpleModelComparison

__all__ = [
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
