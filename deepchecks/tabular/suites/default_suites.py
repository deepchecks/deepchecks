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
"""Functions for loading the default (built-in) suites for various validation stages.

Each function returns a new suite that is initialized with a list of checks and default conditions.
It is possible to customize these suites by editing the checks and conditions inside it after the suites' creation.
"""
import warnings

from deepchecks.tabular import Suite
from deepchecks.tabular.checks import (BoostingOverfit, CalibrationScore, CategoryMismatchTrainTest, ConflictingLabels,
                                       ConfusionMatrixReport, DataDuplicates, DatasetsSizeComparison,
                                       DateTrainTestLeakageDuplicates, DateTrainTestLeakageOverlap,
                                       FeatureLabelCorrelation, FeatureLabelCorrelationChange, IdentifierLeakage,
                                       IndexTrainTestLeakage, IsSingleValue, MixedDataTypes, MixedNulls,
                                       ModelErrorAnalysis, ModelInferenceTime, NewLabelTrainTest,
                                       OutlierSampleDetection, PerformanceReport, RegressionErrorDistribution,
                                       RegressionSystematicError, RocReport, SegmentPerformance, SimpleModelComparison,
                                       SpecialCharacters, StringLengthOutOfBounds, StringMismatch,
                                       StringMismatchComparison, TrainTestFeatureDrift, TrainTestLabelDrift,
                                       TrainTestPredictionDrift, TrainTestSamplesMix, UnusedFeatures, WholeDatasetDrift)

__all__ = ['single_dataset_integrity', 'train_test_leakage', 'train_test_validation',
           'model_evaluation', 'full_suite']


def single_dataset_integrity(**kwargs) -> Suite:
    """
    Create a suite that is meant to detect integrity issues within a single dataset (Deprecated) .

    .. deprecated:: 0.7.0
            `single_dataset_integrity` is deprecated and will be removed in deepchecks 0.8 version, it is replaced by
            `data_integrity` suite.
    """
    warnings.warn(
        'the single_dataset_integrity suite is deprecated, use the data_integrity suite instead',
        DeprecationWarning
    )
    return data_integrity(**kwargs)


def data_integrity(**kwargs) -> Suite:
    """Create a suite that is meant to detect integrity issues within a single dataset."""
    return Suite(
        'Data Integrity Suite',
        IsSingleValue(**kwargs).add_condition_not_single_value(),
        SpecialCharacters(**kwargs).add_condition_ratio_of_special_characters_not_grater_than(),
        MixedNulls(**kwargs).add_condition_different_nulls_not_more_than(),
        MixedDataTypes(**kwargs).add_condition_rare_type_ratio_not_in_range(),
        StringMismatch(**kwargs).add_condition_no_variants(),
        DataDuplicates(**kwargs).add_condition_ratio_not_greater_than(),
        StringLengthOutOfBounds(**kwargs).add_condition_ratio_of_outliers_not_greater_than(),
        ConflictingLabels(**kwargs).add_condition_ratio_of_conflicting_labels_not_greater_than(),
        OutlierSampleDetection(**kwargs),
        FeatureLabelCorrelation(**kwargs).add_condition_feature_pps_not_greater_than()
    )


def train_test_leakage(**kwargs) -> Suite:
    """
    Create a suite that is meant to detect data leakage between the training dataset and the test dataset (Deprecated).

    .. deprecated:: 0.7.0
            `train_test_leakage` is deprecated and will be removed in deepchecks 0.8 version, it is replaced by
            `train_test_validation` suite.
    """
    warnings.warn(
        'the train_test_leakage suite is deprecated, use the train_test_validation suite instead',
        DeprecationWarning
    )
    return train_test_validation(**kwargs)


def train_test_validation(**kwargs) -> Suite:
    """Create a suite that is meant to validate correctness of train-test split, including integrity, \
    distribution and leakage checks."""
    return Suite(
        'Train Test Validation Suite',
        DatasetsSizeComparison(**kwargs).add_condition_test_train_size_ratio_not_smaller_than(),
        NewLabelTrainTest(**kwargs).add_condition_new_labels_not_greater_than(),
        CategoryMismatchTrainTest(**kwargs).add_condition_new_category_ratio_not_greater_than(),
        StringMismatchComparison(**kwargs).add_condition_no_new_variants(),
        DateTrainTestLeakageDuplicates(**kwargs).add_condition_leakage_ratio_not_greater_than(),
        DateTrainTestLeakageOverlap(**kwargs).add_condition_leakage_ratio_not_greater_than(),
        IndexTrainTestLeakage(**kwargs).add_condition_ratio_not_greater_than(),
        IdentifierLeakage(**kwargs).add_condition_pps_not_greater_than(),
        TrainTestSamplesMix(**kwargs).add_condition_duplicates_ratio_not_greater_than(),
        FeatureLabelCorrelationChange(**kwargs).add_condition_feature_pps_difference_not_greater_than()
        .add_condition_feature_pps_in_train_not_greater_than(),
        TrainTestFeatureDrift(**kwargs).add_condition_drift_score_not_greater_than(),
        TrainTestLabelDrift(**kwargs).add_condition_drift_score_not_greater_than(),
        WholeDatasetDrift(**kwargs).add_condition_overall_drift_value_not_greater_than(),
    )


def model_evaluation(**kwargs) -> Suite:
    """Create a suite that is meant to test model performance and overfit."""
    return Suite(
        'Model Evaluation Suite',
        PerformanceReport(**kwargs).add_condition_train_test_relative_degradation_not_greater_than(),
        RocReport(**kwargs).add_condition_auc_not_less_than(),
        ConfusionMatrixReport(**kwargs),
        SegmentPerformance(**kwargs),
        TrainTestPredictionDrift(**kwargs).add_condition_drift_score_not_greater_than(),
        SimpleModelComparison(**kwargs).add_condition_gain_not_less_than(),
        ModelErrorAnalysis(**kwargs).add_condition_segments_performance_relative_difference_not_greater_than(),
        CalibrationScore(**kwargs),
        RegressionSystematicError(**kwargs).add_condition_systematic_error_ratio_to_rmse_not_greater_than(),
        RegressionErrorDistribution(**kwargs).add_condition_kurtosis_not_less_than(),
        UnusedFeatures(**kwargs).add_condition_number_of_high_variance_unused_features_not_greater_than(),
        BoostingOverfit(**kwargs).add_condition_test_score_percent_decline_not_greater_than(),
        ModelInferenceTime(**kwargs).add_condition_inference_time_is_not_greater_than(),
    )


def full_suite(**kwargs) -> Suite:
    """Create a suite that includes many of the implemented checks, for a quick overview of your model and data."""
    return Suite(
        'Full Suite',
        model_evaluation(**kwargs),
        train_test_validation(**kwargs),
        data_integrity(**kwargs),
    )
