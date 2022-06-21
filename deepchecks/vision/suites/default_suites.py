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
from deepchecks.utils.decorators import ParametersCombiner

__all__ = ['train_test_validation', 'model_evaluation', 'full_suite', 'integrity_validation', 'data_integrity']


train_test_validation_kwargs_doc = ParametersCombiner(
    NewLabels,
    SimilarImageLeakage,
    HeatmapComparison,
    TrainTestLabelDrift,
    ImagePropertyDrift,
    ImageDatasetDrift,
    FeatureLabelCorrelationChange,
)


@train_test_validation_kwargs_doc
def train_test_validation(**kwargs) -> Suite:
    """Create a suite that is meant to validate correctness of train-test split, including integrity, \
    distribution and leakage checks.
    
    Parameters
    ----------
    {combined_parameters}
    """
    return Suite(
        'Train Test Validation Suite',
        NewLabels(**kwargs).add_condition_new_label_ratio_less_or_equal(),
        SimilarImageLeakage(**kwargs).add_condition_similar_images_less_or_equal(),
        HeatmapComparison(**kwargs),
        TrainTestLabelDrift(**kwargs).add_condition_drift_score_less_than(),
        ImagePropertyDrift(**kwargs).add_condition_drift_score_less_than(),
        ImageDatasetDrift(**kwargs),
        FeatureLabelCorrelationChange(**kwargs).add_condition_feature_pps_difference_less_than(),
    )


model_evaluation_kwargs_doc = ParametersCombiner(
    ClassPerformance,
    MeanAveragePrecisionReport,
    MeanAverageRecallReport,
    TrainTestPredictionDrift,
    SimpleModelComparison,
    ConfusionMatrixReport,
    ImageSegmentPerformance,
    ModelErrorAnalysis
)


@model_evaluation_kwargs_doc
def model_evaluation(**kwargs) -> Suite:
    """Create a suite that is meant to test model performance and overfit.
    
    Parameters
    ----------
    {combined_parameters}
    """
    return Suite(
        'Model Evaluation Suite',
        ClassPerformance(**kwargs).add_condition_train_test_relative_degradation_less_than(),
        MeanAveragePrecisionReport(**kwargs).add_condition_average_mean_average_precision_greater_than(),
        MeanAverageRecallReport(**kwargs),
        TrainTestPredictionDrift(**kwargs).add_condition_drift_score_less_than(),
        SimpleModelComparison(**kwargs).add_condition_gain_greater_than(),
        ConfusionMatrixReport(**kwargs),
        ImageSegmentPerformance(**kwargs).add_condition_score_from_mean_ratio_greater_than(),
        ModelErrorAnalysis(**kwargs)
    )


data_integrity_kwargs_doc = ParametersCombiner(
    ImagePropertyOutliers,
    LabelPropertyOutliers
)


@data_integrity_kwargs_doc
def integrity_validation(**kwargs) -> Suite:
    """Create a suite that is meant to validate integrity of the data.

    .. deprecated:: 0.7.0
            `integrity_validation` is deprecated and will be removed in deepchecks 0.8 version, it is replaced by
            `data_integrity` suite.

    Parameters
    ----------
    {combined_parameters}
    """
    warnings.warn(
        'the integrity_validation suite is deprecated, use the data_integrity suite instead',
        DeprecationWarning
    )
    return data_integrity(**kwargs)


@data_integrity_kwargs_doc
def data_integrity(**kwargs) -> Suite:
    """Create a suite that includes integrity checks.
    
    Parameters
    ----------
    {combined_parameters}
    """
    return Suite(
        'Data Integrity Suite',
        ImagePropertyOutliers(**kwargs),
        LabelPropertyOutliers(**kwargs)
    )


full_suite_kwargs_doc = ParametersCombiner(
    *train_test_validation_kwargs_doc.routines,
    *model_evaluation_kwargs_doc.routines,
    *data_integrity_kwargs_doc.routines,
)

@full_suite_kwargs_doc
def full_suite(**kwargs) -> Suite:
    """Create a suite that includes many of the implemented checks, for a quick overview of your model and data.
    
    Parameters
    ----------
    {combined_parameters}
    """
    return Suite(
        'Full Suite',
        model_evaluation(**kwargs),
        train_test_validation(**kwargs),
        data_integrity(**kwargs)
    )
