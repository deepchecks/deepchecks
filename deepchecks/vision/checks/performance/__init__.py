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
"""Module containing the performance check in the vision package."""
from .class_performance import ClassPerformance
from .mean_average_precision_report import MeanAveragePrecisionReport
from .mean_average_recall_report import MeanAverageRecallReport
from .robustness_report import RobustnessReport
from .simple_model_comparison import SimpleModelComparison

__all__ = [
    "ClassPerformance",
    "MeanAveragePrecisionReport",
    "MeanAverageRecallReport",
    "RobustnessReport",
    "SimpleModelComparison"
]
