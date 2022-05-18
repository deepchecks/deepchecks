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
from deepchecks.tabular.checks import (BoostingOverfit, CalibrationScore,
                                       CategoryMismatchTrainTest, ColumnsInfo,
                                       ConflictingLabels,
                                       ConfusionMatrixReport, DataDuplicates,
                                       DatasetsSizeComparison,
                                       DateTrainTestLeakageDuplicates,
                                       DateTrainTestLeakageOverlap,
                                       DominantFrequencyChange,
                                       IdentifierLeakage,
                                       IndexTrainTestLeakage, IsSingleValue,
                                       MixedDataTypes, MixedNulls,
                                       ModelErrorAnalysis, ModelInferenceTime,
                                       ModelInfo, NewLabelTrainTest,
                                       OutlierSampleDetection,
                                       PerformanceReport,
                                       RegressionErrorDistribution,
                                       RegressionSystematicError, RocReport,
                                       SimpleModelComparison,
                                       SingleFeatureContributionTrainTest,
                                       SpecialCharacters,
                                       StringLengthOutOfBounds, StringMismatch,
                                       StringMismatchComparison,
                                       TrainTestFeatureDrift,
                                       TrainTestLabelDrift,
                                       TrainTestSamplesMix, UnusedFeatures,
                                       WholeDatasetDrift,
                                       SingleFeatureContribution)

__all__ = ['single_dataset_integrity', 'train_test_leakage', 'train_test_validation',
           'model_evaluation', 'full_suite']


def single_dataset_integrity() -> Suite:
    """
    Create a suite that is meant to detect integrity issues within a single dataset.

    .. deprecated:: 0.7.0
            `single_dataset_integrity` is deprecated and will be removed in deepchecks 0.8 version, it is replaced by
            `data_integrity` suite.
    """
    warnings.warn(
        'the single_dataset_integrity suite is deprecated, use the data_integrity suite instead',
        DeprecationWarning
    )
    return data_integrity()


def data_integrity() -> Suite:
    """Create a suite that is meant to detect integrity issues within a single dataset."""
    return Suite(
        'Data Integrity Suite',
        IsSingleValue().add_condition_not_single_value(),
        MixedNulls().add_condition_different_nulls_not_more_than(),
        MixedDataTypes().add_condition_rare_type_ratio_not_in_range(),
        StringMismatch().add_condition_no_variants(),
        DataDuplicates().add_condition_ratio_not_greater_than(),
        StringLengthOutOfBounds().add_condition_ratio_of_outliers_not_greater_than(),
        SpecialCharacters().add_condition_ratio_of_special_characters_not_grater_than(),
        ConflictingLabels().add_condition_ratio_of_conflicting_labels_not_greater_than(),
        OutlierSampleDetection(),
        SingleFeatureContribution().add_condition_feature_pps_not_greater_than()
    )


def train_test_leakage() -> Suite:
    """
    Create a suite that is meant to detect data leakage between the training dataset and the test dataset.

    .. deprecated:: 0.7.0
            `train_test_leakage` is deprecated and will be removed in deepchecks 0.8 version, it is replaced by
            `train_test_validation` suite.
    """
    warnings.warn(
        'the train_test_leakage suite is deprecated, use the train_test_validation suite instead',
        DeprecationWarning
    )
    return data_integrity()


def train_test_validation() -> Suite:
    """Create a suite that is meant to validate correctness of train-test split, including integrity, \
    distribution and leakage checks."""
    return Suite(
        'Train Test Validation Suite',
        CategoryMismatchTrainTest().add_condition_new_category_ratio_not_greater_than(),
        DatasetsSizeComparison().add_condition_test_train_size_ratio_not_smaller_than(),
        DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_not_greater_than(),
        DateTrainTestLeakageOverlap().add_condition_leakage_ratio_not_greater_than(),
        DominantFrequencyChange().add_condition_ratio_of_change_not_greater_than(),
        IdentifierLeakage().add_condition_pps_not_greater_than(),
        IndexTrainTestLeakage().add_condition_ratio_not_greater_than(),
        NewLabelTrainTest().add_condition_new_labels_not_greater_than(),
        SingleFeatureContributionTrainTest().add_condition_feature_pps_difference_not_greater_than()
        .add_condition_feature_pps_in_train_not_greater_than(),
        StringMismatchComparison().add_condition_no_new_variants(),
        TrainTestFeatureDrift().add_condition_drift_score_not_greater_than(),
        TrainTestLabelDrift().add_condition_drift_score_not_greater_than(),
        TrainTestSamplesMix().add_condition_duplicates_ratio_not_greater_than(),
        WholeDatasetDrift().add_condition_overall_drift_value_not_greater_than(),
    )


def model_evaluation() -> Suite:
    """Create a suite that is meant to test model performance and overfit."""
    return Suite(
        'Model Evaluation Suite',
        BoostingOverfit().add_condition_test_score_percent_decline_not_greater_than(),
        CalibrationScore(),
        ConfusionMatrixReport(),
        ModelErrorAnalysis().add_condition_segments_performance_relative_difference_not_greater_than(),
        ModelInferenceTime().add_condition_inference_time_is_not_greater_than(),
        PerformanceReport().add_condition_train_test_relative_degradation_not_greater_than(),
        RegressionSystematicError().add_condition_systematic_error_ratio_to_rmse_not_greater_than(),
        RegressionErrorDistribution().add_condition_kurtosis_not_less_than(),
        RocReport().add_condition_auc_not_less_than(),
        # SegmentPerformance(),
        SimpleModelComparison().add_condition_gain_not_less_than(),
        # TrainTestPredictionDrift().add_condition_drift_score_not_greater_than(),
        UnusedFeatures().add_condition_number_of_high_variance_unused_features_not_greater_than(),
    )


def full_suite() -> Suite:
    """Create a suite that includes many of the implemented checks, for a quick overview of your model and data."""
    return Suite(
        'Full Suite',
        ModelInfo(),
        ColumnsInfo(),
        model_evaluation(),
        train_test_validation(),
        data_integrity(),
    )
