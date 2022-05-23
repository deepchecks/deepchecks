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
"""Module containing the performance check in the vision package.

.. deprecated:: 0.7.0
        `deepchecks.vision.checks.performance is deprecated and will be removed in deepchecks 0.8 version.
        Use `deepchecks.vision.checks.model_evaluation` instead.
"""
import warnings

from ..model_evaluation import (ClassPerformance, ConfusionMatrixReport, ImageSegmentPerformance,
                                MeanAveragePrecisionReport, MeanAverageRecallReport, ModelErrorAnalysis,
                                RobustnessReport, SimpleModelComparison)

__all__ = [
    "ClassPerformance",
    "MeanAveragePrecisionReport",
    "MeanAverageRecallReport",
    "RobustnessReport",
    "SimpleModelComparison",
    "ConfusionMatrixReport",
    "ModelErrorAnalysis",
    "ImageSegmentPerformance",
]

warnings.warn(
    "deepchecks.vision.checks.performance is deprecated. Use deepchecks.vision.checks.model_evaluation instead.",
    DeprecationWarning
)
