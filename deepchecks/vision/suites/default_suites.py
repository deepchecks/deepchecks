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
import warnings

from deepchecks.vision import Suite
from deepchecks.vision.checks import (ClassPerformance, ConfusionMatrixReport, FeatureLabelCorrelationChange,
                                      HeatmapComparison, ImageDatasetDrift, ImagePropertyDrift, ImagePropertyOutliers,
                                      ImageSegmentPerformance, LabelPropertyOutliers, MeanAveragePrecisionReport,
                                      MeanAverageRecallReport, ModelErrorAnalysis, NewLabels, SimilarImageLeakage,
                                      SimpleModelComparison, TrainTestLabelDrift, TrainTestPredictionDrift)

__all__ = ['train_test_validation', 'model_evaluation', 'full_suite', 'integrity_validation', 'data_integrity']


def train_test_validation(**kwargs) -> Suite:
    """Create a suite that is meant to validate correctness of train-test split, including integrity, \
    distribution and leakage checks."""
    return Suite(
        'Train Test Validation Suite',
        NewLabels(**kwargs).add_condition_new_label_ratio_not_greater_than(),
        SimilarImageLeakage(**kwargs).add_condition_similar_images_not_more_than(),
        HeatmapComparison(**kwargs),
        TrainTestLabelDrift(**kwargs).add_condition_drift_score_not_greater_than(),
        ImagePropertyDrift(**kwargs).add_condition_drift_score_not_greater_than(),
        ImageDatasetDrift(**kwargs),
        FeatureLabelCorrelationChange(**kwargs).add_condition_feature_pps_difference_not_greater_than(),
    )


def model_evaluation(**kwargs) -> Suite:
    """Create a suite that is meant to test model performance and overfit."""
    return Suite(
        'Model Evaluation Suite',
        ClassPerformance(**kwargs).add_condition_train_test_relative_degradation_not_greater_than(),
        MeanAveragePrecisionReport(**kwargs).add_condition_average_mean_average_precision_not_less_than(),
        MeanAverageRecallReport(**kwargs),
        TrainTestPredictionDrift(**kwargs).add_condition_drift_score_not_greater_than(),
        SimpleModelComparison(**kwargs).add_condition_gain_not_less_than(),
        ConfusionMatrixReport(**kwargs),
        ImageSegmentPerformance(**kwargs).add_condition_score_from_mean_ratio_not_less_than(),
        ModelErrorAnalysis(**kwargs)
    )


def integrity_validation(**kwargs) -> Suite:
    """Create a suite that is meant to validate integrity of the data.

    .. deprecated:: 0.7.0
            `integrity_validation` is deprecated and will be removed in deepchecks 0.8 version, it is replaced by
            `data_integrity` suite.
    """
    warnings.warn(
        'the integrity_validation suite is deprecated, use the data_integrity suite instead',
        DeprecationWarning
    )
    return data_integrity(**kwargs)


def data_integrity(**kwargs) -> Suite:
    """Create a suite that includes integrity checks."""
    return Suite(
        'Data Integrity Suite',
        ImagePropertyOutliers(**kwargs),
        LabelPropertyOutliers(**kwargs)
    )


def full_suite(**kwargs) -> Suite:
    """Create a suite that includes many of the implemented checks, for a quick overview of your model and data."""
    return Suite(
        'Full Suite',
        model_evaluation(**kwargs),
        train_test_validation(**kwargs),
        data_integrity(**kwargs)
    )
