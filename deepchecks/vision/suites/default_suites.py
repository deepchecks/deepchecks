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
"""Functions for loading the default (built-in) vision suites for various validation stages.

Each function returns a new suite that is initialized with a list of checks and default conditions.
It is possible to customize these suites by editing the checks and conditions inside it after the suites' creation.
"""
from deepchecks.vision.checks import ClassPerformance, TrainTestLabelDrift, MeanAveragePrecisionReport, \
    MeanAverageRecallReport, ImagePropertyDrift, ImageDatasetDrift, SimpleModelComparison, ConfusionMatrixReport, \
    TrainTestPredictionDrift, ImageSegmentPerformance, SimpleFeatureContribution
from deepchecks.vision import Suite


__all__ = ['train_test_validation', 'model_evaluation', 'full_suite']

from deepchecks.vision.checks.distribution import HeatmapComparison


def train_test_validation() -> Suite:
    """Create a suite that is meant to validate correctness of train-test split, including integrity, \
    distribution and leakage checks."""
    return Suite(
        'Train Test Validation Suite',
        HeatmapComparison(),
        TrainTestLabelDrift().add_condition_drift_score_not_greater_than(),
        TrainTestPredictionDrift().add_condition_drift_score_not_greater_than(),
        ImagePropertyDrift().add_condition_drift_score_not_greater_than(),
        ImageDatasetDrift(),
        SimpleFeatureContribution().add_condition_feature_pps_difference_not_greater_than()
    )


def model_evaluation() -> Suite:
    """Create a suite that is meant to test model performance and overfit."""
    return Suite(
        'Model Evaluation Suite',
        ClassPerformance().add_condition_train_test_relative_degradation_not_greater_than(),
        MeanAveragePrecisionReport().add_condition_test_mean_average_precision_not_less_than(),
        MeanAverageRecallReport(),
        SimpleModelComparison(),
        ConfusionMatrixReport(),
        ImageSegmentPerformance().add_condition_score_from_mean_ratio_not_less_than()
    )


def full_suite() -> Suite:
    """Create a suite that includes many of the implemented checks, for a quick overview of your model and data."""
    return Suite(
        'Full Suite',
        model_evaluation(),
        train_test_validation()
    )
