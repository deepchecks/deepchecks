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
# pylint: disable=unused-argument
"""Functions for loading the default (built-in) suites for various validation stages.

Each function returns a new suite that is initialized with a list of checks and default conditions.
It is possible to customize these suites by editing the checks and conditions inside it after the suites' creation.
"""
from typing import Callable, Dict, List, Union

from deepchecks.tabular import Suite
from deepchecks.tabular.checks import (BoostingOverfit, CalibrationScore, ConflictingLabels, ConfusionMatrixReport,
                                       DataDuplicates, DatasetsSizeComparison, DateTrainTestLeakageDuplicates,
                                       DateTrainTestLeakageOverlap, FeatureFeatureCorrelation, FeatureLabelCorrelation,
                                       FeatureLabelCorrelationChange, IdentifierLabelCorrelation, IndexTrainTestLeakage,
                                       IsSingleValue, MixedDataTypes, MixedNulls, ModelInferenceTime, MultivariateDrift,
                                       NewCategoryTrainTest, NewLabelTrainTest, OutlierSampleDetection, PercentOfNulls,
                                       RegressionErrorDistribution, RocReport, SimpleModelComparison,
                                       SingleDatasetPerformance, SpecialCharacters, StringLengthOutOfBounds,
                                       StringMismatch, StringMismatchComparison, TrainTestFeatureDrift,
                                       TrainTestLabelDrift, TrainTestPerformance, TrainTestPredictionDrift,
                                       TrainTestSamplesMix, UnusedFeatures, WeakSegmentsPerformance)
from deepchecks.tabular.utils.task_type import TaskType

__all__ = ['data_integrity', 'train_test_validation', 'model_evaluation', 'production_suite', 'full_suite']

from deepchecks.utils.typing import Hashable


def data_integrity(columns: Union[Hashable, List[Hashable]] = None,
                   ignore_columns: Union[Hashable, List[Hashable]] = None,
                   n_top_columns: int = None,
                   n_samples: int = None,
                   random_state: int = 42,
                   n_to_show: int = 5,
                   **kwargs) -> Suite:
    """Suite for detecting integrity issues within a single dataset.

    List of Checks:
        .. list-table:: List of Checks
           :widths: 50 50
           :header-rows: 1

           * - Check Example
             - API Reference
           * - :ref:`plot_tabular_is_single_value`
             - :class:`~deepchecks.tabular.checks.data_integrity.IsSingleValue`
           * - :ref:`plot_tabular_special_chars`
             - :class:`~deepchecks.tabular.checks.data_integrity.SpecialCharacters`
           * - :ref:`plot_tabular_mixed_nulls`
             - :class:`~deepchecks.tabular.checks.data_integrity.MixedNulls`
           * - :ref:`plot_tabular_mixed_data_types`
             - :class:`~deepchecks.tabular.checks.data_integrity.MixedDataTypes`
           * - :ref:`plot_tabular_string_mismatch`
             - :class:`~deepchecks.tabular.checks.data_integrity.StringMismatch`
           * - :ref:`plot_tabular_data_duplicates`
             - :class:`~deepchecks.tabular.checks.data_integrity.DataDuplicates`
           * - :ref:`plot_tabular_string_length_out_of_bounds`
             - :class:`~deepchecks.tabular.checks.data_integrity.StringLengthOutOfBounds`
           * - :ref:`plot_tabular_conflicting_labels`
             - :class:`~deepchecks.tabular.checks.data_integrity.ConflictingLabels`
           * - :ref:`plot_tabular_outlier_sample_detection`
             - :class:`~deepchecks.tabular.checks.data_integrity.OutlierSampleDetection`
           * - :ref:`plot_tabular_feature_label_correlation`
             - :class:`~deepchecks.tabular.checks.data_integrity.FeatureLabelCorrelation`
           * - :ref:`plot_tabular_identifier_label_correlation`
             - :class:`~deepchecks.tabular.checks.data_integrity.IdentifierLabelCorrelation`
           * - :ref:`plot_tabular_feature_feature_correlation`
                - :class:`~deepchecks.tabular.checks.data_integrity.FeatureFeatureCorrelation`

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be checked. If None, all columns will be checked except the ones in `ignore_columns`.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be ignored. If None, no columns will be ignored.
    n_top_columns : int , optional
        number of columns to show ordered by feature importance (date, index, label are first) (check dependent)
    n_samples : int , default: 1_000_000
        number of samples to use for checks that sample data. If none, using the default n_samples per check.
    random_state : int, default: 42
        random seed for all checks.
    n_to_show : int , default: 5
        number of top results to show (check dependent)
    **kwargs : dict
        additional arguments to pass to the checks.

    Returns
    -------
    Suite
        A suite for detecting integrity issues within a single dataset.

    Examples
    --------
    >>> from deepchecks.tabular.suites import data_integrity
    >>> suite = data_integrity(columns=['a', 'b', 'c'], n_samples=1_000_000)
    >>> result = suite.run()
    >>> result.show()

    See Also
    --------
    :ref:`quick_data_integrity`
    """
    args = locals()
    args.pop('kwargs')
    non_none_args = {k: v for k, v in args.items() if v is not None}
    kwargs = {**non_none_args, **kwargs}

    return Suite(
        'Data Integrity Suite',
        IsSingleValue(**kwargs).add_condition_not_single_value(),
        SpecialCharacters(**kwargs).add_condition_ratio_of_special_characters_less_or_equal(),
        MixedNulls(**kwargs).add_condition_different_nulls_less_equal_to(),
        MixedDataTypes(**kwargs).add_condition_rare_type_ratio_not_in_range(),
        StringMismatch(**kwargs).add_condition_no_variants(),
        DataDuplicates(**kwargs).add_condition_ratio_less_or_equal(),
        StringLengthOutOfBounds(**kwargs).add_condition_ratio_of_outliers_less_or_equal(),
        ConflictingLabels(**kwargs).add_condition_ratio_of_conflicting_labels_less_or_equal(),
        OutlierSampleDetection(**kwargs),
        FeatureLabelCorrelation(**kwargs).add_condition_feature_pps_less_than(),
        FeatureFeatureCorrelation(**kwargs).add_condition_max_number_of_pairs_above_threshold(),
        IdentifierLabelCorrelation(**kwargs).add_condition_pps_less_or_equal()
    )


def train_test_validation(columns: Union[Hashable, List[Hashable]] = None,
                          ignore_columns: Union[Hashable, List[Hashable]] = None,
                          n_top_columns: int = None,
                          n_samples: int = None,
                          random_state: int = 42,
                          n_to_show: int = 5,
                          **kwargs) -> Suite:
    """Suite for validating correctness of train-test split, including distribution, \
    leakage and integrity checks.

    List of Checks:
        .. list-table:: List of Checks
           :widths: 50 50
           :header-rows: 1

           * - Check Example
             - API Reference
           * - :ref:`plot_tabular_datasets_size_comparison`
             - :class:`~deepchecks.tabular.checks.train_test_validation.DatasetsSizeComparison`
           * - :ref:`plot_tabular_new_label`
             - :class:`~deepchecks.tabular.checks.train_test_validation.NewLabelTrainTest`
           * - :ref:`plot_tabular_new_category`
             - :class:`~deepchecks.tabular.checks.train_test_validation.CategoryMismatchTrainTest`
           * - :ref:`plot_tabular_string_mismatch_comparison`
             - :class:`~deepchecks.tabular.checks.train_test_validation.StringMismatchComparison`
           * - :ref:`plot_tabular_date_train_test_validation_leakage_duplicates`
             - :class:`~deepchecks.tabular.checks.train_test_validation.DateTrainTestLeakageDuplicates`
           * - :ref:`plot_tabular_date_train_test_validation_leakage_overlap`
             - :class:`~deepchecks.tabular.checks.train_test_validation.DateTrainTestLeakageOverlap`
           * - :ref:`plot_tabular_index_leakage`
             - :class:`~deepchecks.tabular.checks.train_test_validation.IndexTrainTestLeakage`
           * - :ref:`plot_tabular_train_test_samples_mix`
             - :class:`~deepchecks.tabular.checks.train_test_validation.TrainTestSamplesMix`
           * - :ref:`plot_tabular_feature_label_correlation_change`
             - :class:`~deepchecks.tabular.checks.train_test_validation.FeatureLabelCorrelationChange`
           * - :ref:`plot_tabular_train_test_feature_drift`
             - :class:`~deepchecks.tabular.checks.train_test_validation.TrainTestFeatureDrift`
           * - :ref:`plot_tabular_train_test_label_drift`
             - :class:`~deepchecks.tabular.checks.train_test_validation.TrainTestLabelDrift`
           * - :ref:`plot_tabular_multivariate_drift`
             - :class:`~deepchecks.tabular.checks.train_test_validation.MultivariateDrift`

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be checked. If None, all columns will be checked except the ones in `ignore_columns`.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be ignored. If None, no columns will be ignored.
    n_top_columns : int , optional
        number of columns to show ordered by feature importance (date, index, label are first) (check dependent)
    n_samples : int , default: None
        number of samples to use for checks that sample data. If none, using the default n_samples per check.
    random_state : int, default: 42
        random seed for all checkss.
    n_to_show : int , default: 5
        number of top results to show (check dependent)
    **kwargs : dict
        additional arguments to pass to the checks.

    Returns
    -------
    Suite
        A suite for validating correctness of train-test split, including distribution, \
        leakage and integrity checks.

    Examples
    --------
    >>> from deepchecks.tabular.suites import train_test_validation
    >>> suite = train_test_validation(columns=['a', 'b', 'c'], n_samples=1_000_000)
    >>> result = suite.run()
    >>> result.show()

    See Also
    --------
    :ref:`quick_train_test_validation`
    """
    args = locals()
    args.pop('kwargs')
    non_none_args = {k: v for k, v in args.items() if v is not None}
    kwargs = {**non_none_args, **kwargs}
    return Suite(
        'Train Test Validation Suite',
        DatasetsSizeComparison(**kwargs).add_condition_test_train_size_ratio_greater_than(),
        NewLabelTrainTest(**kwargs).add_condition_new_labels_number_less_or_equal(),
        NewCategoryTrainTest(**kwargs).add_condition_new_category_ratio_less_or_equal(),
        StringMismatchComparison(**kwargs).add_condition_no_new_variants(),
        DateTrainTestLeakageDuplicates(**kwargs).add_condition_leakage_ratio_less_or_equal(),
        DateTrainTestLeakageOverlap(**kwargs).add_condition_leakage_ratio_less_or_equal(),
        IndexTrainTestLeakage(**kwargs).add_condition_ratio_less_or_equal(),
        TrainTestSamplesMix(**kwargs).add_condition_duplicates_ratio_less_or_equal(),
        FeatureLabelCorrelationChange(**kwargs).add_condition_feature_pps_difference_less_than()
        .add_condition_feature_pps_in_train_less_than(),
        TrainTestFeatureDrift(**kwargs).add_condition_drift_score_less_than(),
        TrainTestLabelDrift(**kwargs).add_condition_drift_score_less_than(),
        MultivariateDrift(**kwargs).add_condition_overall_drift_value_less_than(),
    )


def model_evaluation(alternative_scorers: Dict[str, Callable] = None,
                     columns: Union[Hashable, List[Hashable]] = None,
                     ignore_columns: Union[Hashable, List[Hashable]] = None,
                     n_top_columns: int = None,
                     n_samples: int = None,
                     random_state: int = 42,
                     n_to_show: int = 5,
                     **kwargs) -> Suite:
    """Suite for evaluating the model's performance over different metrics, segments, error analysis, examining \
       overfitting, comparing to baseline, and more.

    List of Checks:
        .. list-table:: List of Checks
           :widths: 50 50
           :header-rows: 1

           * - Check Example
             - API Reference
           * - :ref:`plot_tabular_roc_report`
             - :class:`~deepchecks.tabular.checks.model_evaluation.RocReport`
           * - :ref:`plot_tabular_confusion_matrix_report`
             - :class:`~deepchecks.tabular.checks.model_evaluation.ConfusionMatrixReport`
           * - :ref:`plot_tabular_weak_segment_performance`
             - :class:`~deepchecks.tabular.checks.model_evaluation.WeakSegmentPerformance`
           * - :ref:`plot_tabular_train_test_prediction_drift`
             - :class:`~deepchecks.tabular.checks.model_evaluation.TrainTestPredictionDrift`
           * - :ref:`plot_tabular_simple_model_comparison`
             - :class:`~deepchecks.tabular.checks.model_evaluation.SimpleModelComparison`
           * - :ref:`plot_tabular_calibration_score`
             - :class:`~deepchecks.tabular.checks.model_evaluation.CalibrationScore`
           * - :ref:`plot_tabular_regression_systematic_error`
             - :class:`~deepchecks.tabular.checks.model_evaluation.RegressionSystematicError`
           * - :ref:`plot_tabular_regression_error_distribution`
             - :class:`~deepchecks.tabular.checks.model_evaluation.RegressionErrorDistribution`
           * - :ref:`plot_tabular_unused_features`
             - :class:`~deepchecks.tabular.checks.model_evaluation.UnusedFeatures`
           * - :ref:`plot_tabular_boosting_overfit`
             - :class:`~deepchecks.tabular.checks.model_evaluation.BoostingOverfit`
           * - :ref:`plot_tabular_model_inference_time`
             - :class:`~deepchecks.tabular.checks.model_evaluation.ModelInferenceTime`
           * - :ref:`plot_tabular_train_test_prediction_drift`
                - :class:`~deepchecks.tabular.checks.model_evaluation.TrainTestPredictionDrift`

    Parameters
    ----------
    alternative_scorers : Dict[str, Callable], default: None
        An optional dictionary of scorer name to scorer functions.
        If none given, use default scorers
    columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be checked. If None, all columns will be checked except the ones in `ignore_columns`.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be ignored. If None, no columns will be ignored.
    n_top_columns : int , optional
        number of columns to show ordered by feature importance (date, index, label are first) (check dependent)
    n_samples : int , default: 1_000_000
        number of samples to use for checks that sample data. If none, use the default n_samples per check.
    random_state : int, default: 42
        random seed for all checks.
    n_to_show : int , default: 5
        number of top results to show (check dependent)
    **kwargs : dict
        additional arguments to pass to the checks.

    Returns
    -------
    Suite
        A suite for evaluating the model's performance.

    Examples
    --------
    >>> from deepchecks.tabular.suites import model_evaluation
    >>> suite = model_evaluation(columns=['a', 'b', 'c'], n_samples=1_000_000)
    >>> result = suite.run()
    >>> result.show()

    See Also
    --------
    :ref:`quick_full_suite`
    """
    args = locals()
    args.pop('kwargs')
    non_none_args = {k: v for k, v in args.items() if v is not None}
    kwargs = {**non_none_args, **kwargs}

    return Suite(
        'Model Evaluation Suite',
        TrainTestPerformance(**kwargs).add_condition_train_test_relative_degradation_less_than(),
        RocReport(**kwargs).add_condition_auc_greater_than(),
        ConfusionMatrixReport(**kwargs),
        TrainTestPredictionDrift(**kwargs).add_condition_drift_score_less_than(),
        SimpleModelComparison(**kwargs).add_condition_gain_greater_than(),
        WeakSegmentsPerformance(**kwargs).add_condition_segments_relative_performance_greater_than(),
        CalibrationScore(**kwargs),
        RegressionErrorDistribution(
            **kwargs).add_condition_kurtosis_greater_than().add_condition_systematic_error_ratio_to_rmse_less_than(),
        UnusedFeatures(**kwargs).add_condition_number_of_high_variance_unused_features_less_or_equal(),
        BoostingOverfit(**kwargs).add_condition_test_score_percent_decline_less_than(),
        ModelInferenceTime(**kwargs).add_condition_inference_time_less_than(),
    )


def production_suite(task_type: str = None,
                     is_comparative: bool = True,
                     alternative_scorers: Dict[str, Callable] = None,
                     columns: Union[Hashable, List[Hashable]] = None,
                     ignore_columns: Union[Hashable, List[Hashable]] = None,
                     n_top_columns: int = None,
                     n_samples: int = None,
                     random_state: int = 42,
                     n_to_show: int = 5,
                     **kwargs) -> Suite:
    """Suite for testing the model in production.

    The suite contains checks for evaluating the model's performance. Checks for detecting drift and checks for data
    integrity issues that may occur in production.

    List of Checks (exact checks depend on the task type and the is_comparative flag):
        .. list-table:: List of Checks
           :widths: 50 50
           :header-rows: 1

           * - Check Example
             - API Reference
           * - :ref:`plot_tabular_roc_report`
             - :class:`~deepchecks.tabular.checks.model_evaluation.RocReport`
           * - :ref:`plot_tabular_confusion_matrix_report`
             - :class:`~deepchecks.tabular.checks.model_evaluation.ConfusionMatrixReport`
           * - :ref:`plot_tabular_weak_segment_performance`
             - :class:`~deepchecks.tabular.checks.model_evaluation.WeakSegmentPerformance`
           * - :ref:`plot_tabular_regression_error_distribution`
             - :class:`~deepchecks.tabular.checks.model_evaluation.RegressionErrorDistribution`
           * - :ref:`plot_tabular_string_mismatch_comparison`
             - :class:`~deepchecks.tabular.checks.train_test_validation.StringMismatchComparison`
           * - :ref:`plot_tabular_feature_label_correlation_change`
             - :class:`~deepchecks.tabular.checks.train_test_validation.FeatureLabelCorrelationChange`
           * - :ref:`plot_tabular_train_test_feature_drift`
             - :class:`~deepchecks.tabular.checks.train_test_validation.TrainTestFeatureDrift`
           * - :ref:`plot_tabular_train_test_label_drift`
             - :class:`~deepchecks.tabular.checks.train_test_validation.TrainTestLabelDrift`
           * - :ref:`plot_tabular_multivariate_drift`
             - :class:`~deepchecks.tabular.checks.train_test_validation.MultivariateDrift`
           * - :ref:`plot_tabular_train_test_prediction_drift`
             - :class:`~deepchecks.tabular.checks.model_evaluation.TrainTestPredictionDrift`
           * - :ref:`plot_tabular_train_test_prediction_drift`
             - :class:`~deepchecks.tabular.checks.model_evaluation.TrainTestPredictionDrift`
           * - :ref:`plot_tabular_string_mismatch`
             - :class:`~deepchecks.tabular.checks.data_integrity.StringMismatch`
           * - :ref:`plot_tabular_feature_label_correlation`
             - :class:`~deepchecks.tabular.checks.data_integrity.FeatureLabelCorrelation`
           * - :ref:`plot_tabular_feature_feature_correlation`
             - :class:`~deepchecks.tabular.checks.data_integrity.FeatureFeatureCorrelation`
           * - :ref:`plot_tabular_single_dataset_performance`
             - :class:`~deepchecks.tabular.checks.model_evaluation.SingleDatasetPerformance`

    Parameters
    ----------
    task_type : str, default: None
        The type of the task. Must be one of 'binary', 'multiclass' or 'regression'. If not given, both checks for
        classification and regression will be added to the suite.
    is_comparative : bool, default: True
        Whether to add the checks comparing the production data to some reference data, or if False, to add the
        checks inspecting the production data only.
    alternative_scorers : Dict[str, Callable], default: None
        An optional dictionary of scorer name to scorer functions.
        If none given, use default scorers
    columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be checked. If None, all columns will be checked except the ones in `ignore_columns`.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be ignored. If None, no columns will be ignored.
    n_top_columns : int , optional
        number of columns to show ordered by feature importance (date, index, label are first) (check dependent)
    n_samples : int , default: 1_000_000
        number of samples to use for checks that sample data. If none, use the default n_samples per check.
    random_state : int, default: 42
        random seed for all checks.
    n_to_show : int , default: 5
        number of top results to show (check dependent)
    **kwargs : dict
        additional arguments to pass to the checks.

    Returns
    -------
    Suite
        A suite for evaluating the model's performance.

    Examples
    --------
    >>> from deepchecks.tabular.suites import production_suite
    >>> suite = production_suite(task_type='binary', n_samples=10_000)
    >>> result = suite.run()
    >>> result.show()

    See Also
    --------
    :ref:`quick_full_suite`
    """
    args = locals()
    args.pop('kwargs')
    non_none_args = {k: v for k, v in args.items() if v is not None}
    kwargs = {**non_none_args, **kwargs}

    checks = [WeakSegmentsPerformance(**kwargs).add_condition_segments_relative_performance_greater_than(),
              PercentOfNulls(**kwargs)]

    # Add checks for regression and classification
    regression_checks = [RegressionErrorDistribution(**kwargs).add_condition_kurtosis_greater_than()]
    classification_checks = [ConfusionMatrixReport(**kwargs), RocReport(**kwargs).add_condition_auc_greater_than()]
    if task_type is None:
        checks.extend(classification_checks)
        checks.extend(regression_checks)
    elif task_type == TaskType.REGRESSION.value:
        checks.extend(regression_checks)
    else:
        checks.extend(classification_checks)

    if is_comparative:
        checks.append(StringMismatchComparison(**kwargs).add_condition_no_new_variants())
        checks.append(FeatureLabelCorrelationChange(**kwargs).add_condition_feature_pps_difference_less_than())
        checks.append(TrainTestFeatureDrift(**kwargs).add_condition_drift_score_less_than())
        checks.append(MultivariateDrift(**kwargs).add_condition_overall_drift_value_less_than())
        checks.append(TrainTestLabelDrift(ignore_na=True, **kwargs).add_condition_drift_score_less_than())
        checks.append(TrainTestPredictionDrift(**kwargs).add_condition_drift_score_less_than())
        checks.append(TrainTestPerformance(**kwargs).add_condition_train_test_relative_degradation_less_than())
    else:
        checks.append(StringMismatch(**kwargs).add_condition_no_variants())
        checks.append(FeatureLabelCorrelation(**kwargs).add_condition_feature_pps_less_than())
        checks.append(FeatureFeatureCorrelation(**kwargs).add_condition_max_number_of_pairs_above_threshold())
        checks.append(SingleDatasetPerformance(**kwargs))

    return Suite('Production Suite', *checks)


def full_suite(**kwargs) -> Suite:
    """Create a suite that includes many of the implemented checks, for a quick overview of your model and data."""
    return Suite(
        'Full Suite',
        model_evaluation(**kwargs),
        train_test_validation(**kwargs),
        data_integrity(**kwargs),
    )
