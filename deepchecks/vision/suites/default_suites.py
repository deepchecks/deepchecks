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
from deepchecks.vision import Suite
from deepchecks.vision.checks import (ClassPerformance, ConfusionMatrixReport,
                                      HeatmapComparison, ImageDatasetDrift,
                                      ImagePropertyDrift,
                                      ImagePropertyOutliers,
                                      ImageSegmentPerformance,
                                      LabelPropertyOutliers,
                                      MeanAveragePrecisionReport,
                                      MeanAverageRecallReport,
                                      ModelErrorAnalysis, NewLabels,
                                      SimilarImageLeakage,
                                      SingleFeatureContribution,
                                      SimpleModelComparison,
                                      TrainTestLabelDrift,
                                      TrainTestPredictionDrift)

__all__ = ['train_test_validation', 'model_evaluation', 'full_suite', 'integrity_validation']


def train_test_validation(**kwargs) -> Suite:
    """Create a suite that is meant to validate correctness of train-test split, including integrity, \
    distribution and leakage checks."""
    return Suite(
        'Train Test Validation Suite',
        SimilarImageLeakage(**kwargs).add_condition_similar_images_not_more_than(),
        HeatmapComparison(**kwargs),
        TrainTestLabelDrift(**kwargs).add_condition_drift_score_not_greater_than(),
        TrainTestPredictionDrift(**kwargs).add_condition_drift_score_not_greater_than(),
        ImagePropertyDrift(**kwargs).add_condition_drift_score_not_greater_than(),
        ImageDatasetDrift(**kwargs),
        SingleFeatureContribution(**kwargs).add_condition_feature_pps_difference_not_greater_than(),
        NewLabels(**kwargs).add_condition_new_label_ratio_not_greater_than()
    )


def model_evaluation(**kwargs) -> Suite:
    """Create a suite that is meant to test model performance and overfit."""
    return Suite(
        'Model Evaluation Suite',
        ClassPerformance(**kwargs).add_condition_train_test_relative_degradation_not_greater_than(),
        MeanAveragePrecisionReport(**kwargs).add_condition_average_mean_average_precision_not_less_than(),
        MeanAverageRecallReport(**kwargs),
        SimpleModelComparison(**kwargs).add_condition_gain_not_less_than(),
        ConfusionMatrixReport(**kwargs),
        ImageSegmentPerformance(**kwargs).add_condition_score_from_mean_ratio_not_less_than(),
        ModelErrorAnalysis(**kwargs)
    )


def integrity_validation(**kwargs) -> Suite:
    """Create a suite that is meant to validate integrity of the data."""
    return Suite(
        'Integrity Validation Suite',
        ImagePropertyOutliers(**kwargs),
        LabelPropertyOutliers(**kwargs)
    )


def full_suite(**kwargs) -> Suite:
    """Create a suite that includes many of the implemented checks, for a quick overview of your model and data."""
    return Suite(
        'Full Suite',
        model_evaluation(**kwargs),
        train_test_validation(**kwargs),
        integrity_validation(**kwargs)
    )
