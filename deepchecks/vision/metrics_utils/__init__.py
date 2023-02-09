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
"""Module containing metrics utils."""

from .confusion_matrix_counts_metrics import AVAILABLE_EVALUATING_FUNCTIONS
from .custom_metric import CustomMetric
from .custom_scorer import CustomClassificationScorer
from .detection_precision_recall import AveragePrecisionRecall, ObjectDetectionAveragePrecision
from .detection_tp_fp_fn_calc import ObjectDetectionTpFpFn, TpFpFn
from .scorers import get_scorers_dict, metric_results_to_df

__all__ = [
    'get_scorers_dict',
    'metric_results_to_df',
    'ObjectDetectionAveragePrecision',
    'AveragePrecisionRecall',
    'ObjectDetectionTpFpFn',
    'TpFpFn',
    'AVAILABLE_EVALUATING_FUNCTIONS',
    'CustomClassificationScorer',
    'CustomMetric'
]
