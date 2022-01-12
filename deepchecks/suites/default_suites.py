# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Predefined suites for various use-cases."""
from deepchecks.checks import (
    MixedNulls, SpecialCharacters, StringLengthOutOfBounds, StringMismatch, MixedDataTypes,
    DateTrainTestLeakageDuplicates, SingleFeatureContributionTrainTest, TrainTestSamplesMix,
    DateTrainTestLeakageOverlap, IdentifierLeakage, IndexTrainTestLeakage, DominantFrequencyChange,
    CategoryMismatchTrainTest, NewLabelTrainTest, StringMismatchComparison, TrainTestFeatureDrift, WholeDatasetDrift,
    ConfusionMatrixReport, RocReport, CalibrationScore, TrustScoreComparison,
    RegressionErrorDistribution, RegressionSystematicError, PerformanceReport, SimpleModelComparison, BoostingOverfit,
    ModelInfo, ColumnsInfo, DataDuplicates, IsSingleValue, LabelAmbiguity,
    DatasetsSizeComparison, UnusedFeatures, ModelInferenceTimeCheck, ModelErrorAnalysis, TrainTestLabelDrift
)
from deepchecks import Suite

__all__ = ['single_dataset_integrity', 'train_test_leakage', 'train_test_validation',
           'model_evaluation', 'full_suite']


def single_dataset_integrity() -> Suite:
    """Create 'Single Dataset Integrity Suite'.

    The suite runs a set of checks that are meant to detect integrity issues within a single dataset.
    """
    return Suite(
        'Single Dataset Integrity Suite',
        IsSingleValue().add_condition_not_single_value(),
        MixedNulls().add_condition_different_nulls_not_more_than(),
        MixedDataTypes().add_condition_rare_type_ratio_not_in_range(),
        StringMismatch().add_condition_no_variants(),
        DataDuplicates().add_condition_ratio_not_greater_than(),
        StringLengthOutOfBounds().add_condition_ratio_of_outliers_not_greater_than(),
        SpecialCharacters().add_condition_ratio_of_special_characters_not_grater_than(),
        LabelAmbiguity().add_condition_ambiguous_sample_ratio_not_greater_than()
    )


def train_test_leakage() -> Suite:
    """Create 'Train Test Leakage Suite'.

    The suite runs a set of checks that are meant to detect data leakage from the training dataset to the test dataset.
    """
    return Suite(
        'Train Test Leakage Suite',
        DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_not_greater_than(),
        DateTrainTestLeakageOverlap().add_condition_leakage_ratio_not_greater_than(),
        SingleFeatureContributionTrainTest().add_condition_feature_pps_difference_not_greater_than()
        .add_condition_feature_pps_in_train_not_greater_than(),
        TrainTestSamplesMix().add_condition_duplicates_ratio_not_greater_than(),
        IdentifierLeakage().add_condition_pps_not_greater_than(),
        IndexTrainTestLeakage().add_condition_ratio_not_greater_than()
    )


def train_test_validation() -> Suite:
    """Create 'Train Test Validation Suite'.

    The suite runs a set of checks that are meant to validate correctness of train-test split, including
    integrity, drift and leakage.
    """
    return Suite(
        'Train Test Validation Suite',
        TrainTestFeatureDrift().add_condition_drift_score_not_greater_than(),
        TrainTestLabelDrift().add_condition_drift_score_not_greater_than(),
        WholeDatasetDrift().add_condition_overall_drift_value_not_greater_than(),
        DominantFrequencyChange().add_condition_ratio_of_change_not_greater_than(),
        CategoryMismatchTrainTest().add_condition_new_category_ratio_not_greater_than(),
        NewLabelTrainTest().add_condition_new_labels_not_greater_than(),
        StringMismatchComparison().add_condition_no_new_variants(),
        DatasetsSizeComparison().add_condition_test_train_size_ratio_not_smaller_than(),
        train_test_leakage()
    )


def model_evaluation() -> Suite:
    """Create 'Model Evaluation Suite'.

    The suite runs a set of checks that are meant to test model performance and overfit.
    """
    return Suite(
        'Model Evaluation Suite',
        ConfusionMatrixReport(),
        PerformanceReport().add_condition_train_test_relative_degradation_not_greater_than(),
        RocReport().add_condition_auc_not_less_than(),
        SimpleModelComparison().add_condition_gain_not_less_than(),
        ModelErrorAnalysis().add_condition_segments_performance_relative_difference_not_greater_than(),
        CalibrationScore(),
        TrustScoreComparison().add_condition_mean_score_percent_decline_not_greater_than(),
        RegressionSystematicError().add_condition_systematic_error_ratio_to_rmse_not_greater_than(),
        RegressionErrorDistribution().add_condition_kurtosis_not_less_than(),
        BoostingOverfit().add_condition_test_score_percent_decline_not_greater_than(),
        UnusedFeatures().add_condition_number_of_high_variance_unused_features_not_greater_than(),
        ModelInferenceTimeCheck().add_condition_inference_time_is_not_greater_than()
    )


def full_suite() -> Suite:
    """Create 'Full Suite'.

    The suite runs all deepchecks' checks.
    """
    return Suite(
        'Full Suite',
        ModelInfo(),
        ColumnsInfo(),
        model_evaluation(),
        train_test_validation(),
        single_dataset_integrity(),
    )
