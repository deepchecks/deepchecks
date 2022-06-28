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
from typing import Callable, Dict, List, Union

from deepchecks.tabular import Suite
from deepchecks.tabular.checks import (BoostingOverfit, CalibrationScore, CategoryMismatchTrainTest, ConflictingLabels,
                                       ConfusionMatrixReport, DataDuplicates, DatasetsSizeComparison,
                                       DateTrainTestLeakageDuplicates, DateTrainTestLeakageOverlap,
                                       FeatureLabelCorrelation, FeatureLabelCorrelationChange,
                                       IdentifierLabelCorrelation, IndexTrainTestLeakage, IsSingleValue, MixedDataTypes,
                                       MixedNulls, ModelErrorAnalysis, ModelInferenceTime, NewLabelTrainTest,
                                       OutlierSampleDetection, PerformanceReport, RegressionErrorDistribution,
                                       RegressionSystematicError, RocReport, SegmentPerformance, SimpleModelComparison,
                                       SpecialCharacters, StringLengthOutOfBounds, StringMismatch,
                                       StringMismatchComparison, TrainTestFeatureDrift, TrainTestLabelDrift,
                                       TrainTestPredictionDrift, TrainTestSamplesMix, UnusedFeatures, WholeDatasetDrift)

__all__ = ['single_dataset_integrity', 'train_test_leakage', 'train_test_validation',
           'model_evaluation', 'full_suite']

from deepchecks.utils.typing import Hashable


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


def data_integrity(columns: Union[Hashable, List[Hashable]] = None,
                   ignore_columns: Union[Hashable, List[Hashable]] = None,
                   n_top_columns: int = None,
                   n_samples: int = 1_000_000,
                   random_state: int = 42,
                   n_to_show: int = 5,
                   **kwargs) -> Suite:
    """Suite for detecting integrity issues within a single dataset.

    List of Checks:
        .. list-table:: List of Checks
           :widths: 50 50
           :header-rows: 1

           * - Check
             - API Reference
           * - IsSingleValue
             - :class:`~deepchecks.tabular.checks.data_integrity.IsSingleValue`
           * - SpecialCharacters
             - :class:`~deepchecks.tabular.checks.data_integrity.SpecialCharacters`
           * - MixedNulls
             - :class:`~deepchecks.tabular.checks.data_integrity.MixedNulls`
           * - MixedDataTypes
             - :class:`~deepchecks.tabular.checks.data_integrity.MixedDataTypes`
           * - StringMismatch
             - :class:`~deepchecks.tabular.checks.data_integrity.StringMismatch`
           * - DataDuplicates
             - :class:`~deepchecks.tabular.checks.data_integrity.DataDuplicates`
           * - StringLengthOutOfBounds
             - :class:`~deepchecks.tabular.checks.data_integrity.StringLengthOutOfBounds`
           * - ConflictingLabels
             - :class:`~deepchecks.tabular.checks.data_integrity.ConflictingLabels`
           * - OutlierSampleDetection
             - :class:`~deepchecks.tabular.checks.data_integrity.OutlierSampleDetection`
           * - FeatureLabelCorrelation
             - :class:`~deepchecks.tabular.checks.data_integrity.FeatureLabelCorrelation`
           * - IdentifierLabelCorrelation
             - :class:`~deepchecks.tabular.checks.data_integrity.IdentifierLabelCorrelation`

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be checked. If None, all columns will be checked except the ones in `ignore_columns`.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be ignored. If None, no columns will be ignored.
    n_top_columns : int , optional
        amount of columns to show ordered by feature importance (date, index, label are first) (check dependent)
    n_samples : int , default: 1_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    n_to_show : int , default: 5
        number of top results to show (check dependent)
    **kwargs : dict
        additional arguments to pass to the checks.

    Returns
    -------
    Suite
        A suite that is meant to detect integrity issues within a single dataset.

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

    default_kwargs = {'columns': columns, 'ignore_columns': ignore_columns, 'n_top_columns': n_top_columns,
                      'n_samples': n_samples, 'random_state': random_state, 'n_to_show': n_to_show}
    kwargs = {**default_kwargs, **kwargs}

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
        IdentifierLabelCorrelation(**kwargs).add_condition_pps_less_or_equal()
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


def train_test_validation(columns: Union[Hashable, List[Hashable]] = None,
                          ignore_columns: Union[Hashable, List[Hashable]] = None,
                          n_top_columns: int = None,
                          n_samples: int = 1_000_000,
                          random_state: int = 42,
                          n_to_show: int = 5,
                          **kwargs) -> Suite:
    """Suite for validating correctness of train-test split, including integrity, \
    distribution and leakage checks.

    List of Checks:
        .. list-table:: List of Checks
           :widths: 50 50
           :header-rows: 1

           * - Check
             - API Reference
           * - DatasetsSizeComparison
             - :class:`~deepchecks.tabular.checks.train_test_validation.DatasetsSizeComparison`
           * - NewLabelTrainTest
             - :class:`~deepchecks.tabular.checks.train_test_validation.NewLabelTrainTest`
           * - CategoryMismatchTrainTest
             - :class:`~deepchecks.tabular.checks.train_test_validation.CategoryMismatchTrainTest`
           * - StringMismatchComparison
             - :class:`~deepchecks.tabular.checks.train_test_validation.StringMismatchComparison`
           * - DateTrainTestLeakageDuplicates
             - :class:`~deepchecks.tabular.checks.train_test_validation.DateTrainTestLeakageDuplicates`
           * - DateTrainTestLeakageOverlap
             - :class:`~deepchecks.tabular.checks.train_test_validation.DateTrainTestLeakageOverlap`
           * - IndexTrainTestLeakage
             - :class:`~deepchecks.tabular.checks.train_test_validation.IndexTrainTestLeakage`
           * - TrainTestSamplesMix
             - :class:`~deepchecks.tabular.checks.train_test_validation.TrainTestSamplesMix`
           * - FeatureLabelCorrelationChange
             - :class:`~deepchecks.tabular.checks.train_test_validation.FeatureLabelCorrelationChange`
           * - TrainTestFeatureDrift
             - :class:`~deepchecks.tabular.checks.train_test_validation.TrainTestFeatureDrift`
           * - TrainTestLabelDrift
             - :class:`~deepchecks.tabular.checks.train_test_validation.TrainTestLabelDrift`
           * - WholeDatasetDrift
             - :class:`~deepchecks.tabular.checks.train_test_validation.WholeDatasetDrift`

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be checked. If None, all columns will be checked except the ones in `ignore_columns`.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be ignored. If None, no columns will be ignored.
    n_top_columns : int , optional
        amount of columns to show ordered by feature importance (date, index, label are first) (check dependent)
    n_samples : int , default: 1_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    n_to_show : int , default: 5
        number of top results to show (check dependent)
    **kwargs : dict
        additional arguments to pass to the checks.

    Returns
    -------
    Suite
        A suite that is meant to validate correctness of train-test split, including integrity, \
        distribution and leakage checks.

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

    default_kwargs = {'columns': columns, 'ignore_columns': ignore_columns, 'n_top_columns': n_top_columns,
                      'n_samples': n_samples, 'random_state': random_state, 'n_to_show': n_to_show}
    kwargs = {**default_kwargs, **kwargs}
    return Suite(
        'Train Test Validation Suite',
        DatasetsSizeComparison(**kwargs).add_condition_test_train_size_ratio_greater_than(),
        NewLabelTrainTest(**kwargs).add_condition_new_labels_number_less_or_equal(),
        CategoryMismatchTrainTest(**kwargs).add_condition_new_category_ratio_less_or_equal(),
        StringMismatchComparison(**kwargs).add_condition_no_new_variants(),
        DateTrainTestLeakageDuplicates(**kwargs).add_condition_leakage_ratio_less_or_equal(),
        DateTrainTestLeakageOverlap(**kwargs).add_condition_leakage_ratio_less_or_equal(),
        IndexTrainTestLeakage(**kwargs).add_condition_ratio_less_or_equal(),
        TrainTestSamplesMix(**kwargs).add_condition_duplicates_ratio_less_or_equal(),
        FeatureLabelCorrelationChange(**kwargs).add_condition_feature_pps_difference_less_than()
            .add_condition_feature_pps_in_train_less_than(),
        TrainTestFeatureDrift(**kwargs).add_condition_drift_score_less_than(),
        TrainTestLabelDrift(**kwargs).add_condition_drift_score_less_than(),
        WholeDatasetDrift(**kwargs).add_condition_overall_drift_value_less_than(),
    )


def model_evaluation(alternative_scorers: Dict[str, Callable] = None,
                     columns: Union[Hashable, List[Hashable]] = None,
                     ignore_columns: Union[Hashable, List[Hashable]] = None,
                     n_top_columns: int = None,
                     n_samples: int = 1_000_000,
                     random_state: int = 42,
                     n_to_show: int = 5,
                     **kwargs) -> Suite:
    """Suite for testing the model performance and overfitting.

    List of Checks:
        .. list-table:: List of Checks
           :widths: 50 50
           :header-rows: 1

           * - Check
             - API Reference
           * - PerformanceReport
             - :class:`~deepchecks.tabular.checks.model_evaluation.PerformanceReport`
           * - RocReport
             - :class:`~deepchecks.tabular.checks.model_evaluation.RocReport`
           * - ConfusionMatrixReport
             - :class:`~deepchecks.tabular.checks.model_evaluation.ConfusionMatrixReport`
           * - SegmentPerformance
             - :class:`~deepchecks.tabular.checks.model_evaluation.SegmentPerformance`
           * - TrainTestPredictionDrift
             - :class:`~deepchecks.tabular.checks.model_evaluation.TrainTestPredictionDrift`
           * - SimpleModelComparison
             - :class:`~deepchecks.tabular.checks.model_evaluation.SimpleModelComparison`
           * - ModelErrorAnalysis
             - :class:`~deepchecks.tabular.checks.model_evaluation.ModelErrorAnalysis`
           * - CalibrationScore
             - :class:`~deepchecks.tabular.checks.model_evaluation.CalibrationScore`
           * - RegressionSystematicError
             - :class:`~deepchecks.tabular.checks.model_evaluation.RegressionSystematicError`
           * - RegressionErrorDistribution
             - :class:`~deepchecks.tabular.checks.model_evaluation.RegressionErrorDistribution`
           * - UnusedFeatures
             - :class:`~deepchecks.tabular.checks.model_evaluation.UnusedFeatures`
           * - BoostingOverfit
             - :class:`~deepchecks.tabular.checks.model_evaluation.BoostingOverfit`
           * - ModelInferenceTime
             - :class:`~deepchecks.tabular.checks.model_evaluation.ModelInferenceTime`

    Parameters
    ----------
    alternative_scorers : Dict[str, Callable], default: None
        An optional dictionary of scorer name to scorer functions.
        If none given, using default scorers
    columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be checked. If None, all columns will be checked except the ones in `ignore_columns`.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        The columns to be ignored. If None, no columns will be ignored.
    n_top_columns : int , optional
        amount of columns to show ordered by feature importance (date, index, label are first) (check dependent)
    n_samples : int , default: 1_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    n_to_show : int , default: 5
        number of top results to show (check dependent)
    **kwargs : dict
        additional arguments to pass to the checks.

    Returns
    -------
    Suite
        A Suite for testing the model performance and overfitting.

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

    default_kwargs = {'alternative_scorers': alternative_scorers, 'columns': columns, 'ignore_columns': ignore_columns,
                      'n_top_columns': n_top_columns, 'n_samples': n_samples, 'random_state': random_state,
                      'n_to_show': n_to_show}

    kwargs = {**default_kwargs, **kwargs}

    return Suite(
        'Model Evaluation Suite',
        PerformanceReport(**kwargs).add_condition_train_test_relative_degradation_not_greater_than(),
        RocReport(**kwargs).add_condition_auc_greater_than(),
        ConfusionMatrixReport(**kwargs),
        SegmentPerformance(**kwargs),
        TrainTestPredictionDrift(**kwargs).add_condition_drift_score_less_than(),
        SimpleModelComparison(**kwargs).add_condition_gain_greater_than(),
        ModelErrorAnalysis(**kwargs).add_condition_segments_performance_relative_difference_less_than(),
        CalibrationScore(**kwargs),
        RegressionSystematicError(**kwargs).add_condition_systematic_error_ratio_to_rmse_less_than(),
        RegressionErrorDistribution(**kwargs).add_condition_kurtosis_greater_than(),
        UnusedFeatures(**kwargs).add_condition_number_of_high_variance_unused_features_less_or_equal(),
        BoostingOverfit(**kwargs).add_condition_test_score_percent_decline_less_than(),
        ModelInferenceTime(**kwargs).add_condition_inference_time_less_than(),
    )


def full_suite(**kwargs) -> Suite:
    """Create a suite that includes many of the implemented checks, for a quick overview of your model and data."""
    return Suite(
        'Full Suite',
        model_evaluation(**kwargs),
        train_test_validation(**kwargs),
        data_integrity(**kwargs),
    )
