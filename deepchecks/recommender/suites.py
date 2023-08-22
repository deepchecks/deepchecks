# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
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
from deepchecks.recommender.checks import (LabelPopularityDrift, OperationsAmountSegmentPerformance, PopularityBias,
                                           PredictionPopularityDrift, SamplePerformance, ScatterPerformance)
from deepchecks.tabular import Suite
from deepchecks.tabular.checks import (FeatureDrift, FeatureFeatureCorrelation, FeatureLabelCorrelation,
                                       FeatureLabelCorrelationChange, MultivariateDrift, NewCategoryTrainTest,
                                       PercentOfNulls, SingleDatasetPerformance, TrainTestPerformance,
                                       WeakSegmentsPerformance)

__all__ = ['recsys_suite']


def recsys_suite(is_comparative: bool = True,
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
           * - :ref:`plot_tabular_feature_drift`
             - :class:`~deepchecks.tabular.checks.train_test_validation.FeatureDrift`
           * - :ref:`plot_tabular_label_drift`
             - :class:`~deepchecks.tabular.checks.train_test_validation.LabelDrift`
           * - :ref:`plot_tabular_multivariate_drift`
             - :class:`~deepchecks.tabular.checks.train_test_validation.MultivariateDrift`
           * - :ref:`plot_tabular_prediction_drift`
             - :class:`~deepchecks.tabular.checks.model_evaluation.PredictionDrift`
           * - :ref:`plot_tabular_prediction_drift`
             - :class:`~deepchecks.tabular.checks.model_evaluation.PredictionDrift`
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

    checks = [
        WeakSegmentsPerformance(**kwargs).add_condition_segments_relative_performance_greater_than(),
        PercentOfNulls(**kwargs),
        OperationsAmountSegmentPerformance(**kwargs),
        PopularityBias(**kwargs).add_condition_drift_score_less_than(),
        SamplePerformance(**kwargs),
        ScatterPerformance(**kwargs)
    ]

    if is_comparative:
        checks.append(FeatureLabelCorrelationChange(**kwargs).add_condition_feature_pps_difference_less_than())
        checks.append(FeatureDrift(**kwargs).add_condition_drift_score_less_than())
        checks.append(MultivariateDrift(**kwargs).add_condition_overall_drift_value_less_than())
        checks.append(TrainTestPerformance(**kwargs).add_condition_train_test_relative_degradation_less_than())
        checks.append(NewCategoryTrainTest(**kwargs).add_condition_new_category_ratio_less_or_equal())
    else:
        checks.append(FeatureLabelCorrelation(**kwargs).add_condition_feature_pps_less_than())
        checks.append(FeatureFeatureCorrelation(**kwargs).add_condition_max_number_of_pairs_above_threshold())
        checks.append(SingleDatasetPerformance(**kwargs))
        checks.append(PredictionPopularityDrift(**kwargs).add_condition_drift_score_less_than())
        checks.append(LabelPopularityDrift(**kwargs).add_condition_drift_score_less_than())

    return Suite('RecSys Suite', *checks)
