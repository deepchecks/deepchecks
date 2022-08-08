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
"""Module containing a check for computing a scalar performance metric for a single dataset."""
from numbers import Number
from typing import Any, Callable, Dict, List, Union

from ignite.metrics import Metric

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.checks import ReduceMixin
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import Batch, Context, SingleDatasetCheck
from deepchecks.vision.metrics_utils.scorers import get_scorers_dict, metric_results_to_df

__all__ = ['SingleDatasetPerformance']


class SingleDatasetPerformance(SingleDatasetCheck, ReduceMixin):
    """Calculate performance metrics of a given model on a given dataset.

    Parameters
    ----------
    scorers : Union[Dict[str, Union[Metric, Callable, str]], List[Any]] = None,
        An optional dictionary of scorer name to scorer functions.
        If none given, using default scorers

    """

    def __init__(self, scorers: Union[Dict[str, Union[Metric, Callable, str]], List[Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.scorers = scorers

    def initialize_run(self, context: Context, dataset_kind: DatasetKind.TRAIN):
        """Initialize the metric for the check, and validate task type is relevant."""
        self.scorers = get_scorers_dict(context.get_data_by_kind(dataset_kind), self.scorers)

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind.TRAIN):
        """Update the metrics by passing the batch to ignite metric update method."""
        label = batch.labels
        prediction = batch.predictions
        for scorer in self.scorers.values():
            scorer.update((prediction, label))

    def compute(self, context: Context, dataset_kind: DatasetKind.TRAIN) -> CheckResult:
        """Compute the metric result using the ignite metrics compute method and reduce to a scalar."""
        results_dict = {}
        for name, scorer in self.scorers.items():
            result = scorer.compute()
            results_dict[name] = [result] if isinstance(result, Number) else result
        result_df = metric_results_to_df(results_dict, context.get_data_by_kind(dataset_kind))
        display = result_df if context.with_display else None
        return CheckResult(result_df, header='Single Dataset Performance', display=display)

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return the values of the metrics for the dataset provided in a {metric: value} format."""
        result = {row['Metric'] + '_' + str(row['Class Name']): row['Value'] for _, row in
                  check_result.value.iterrows()}
        for key in [key for key in result.keys() if key.endswith('_<NA>')]:
            result[key.replace('_<NA>', '')] = result.pop(key)
        return result

    def add_condition_greater_than(self, threshold: float, metrics: List[str] = None, class_mode: str = 'all'):
        """Add condition - the result is greater than the threshold.

        Parameters
        ----------
        threshold: float
            The threshold that the metrics result should be grater than.

        metrics: List[str]
            The names of the metrics from the check to apply the condition to. If None, runs on all of the metrics that
            were calculated in the check.

        class_mode: str, default: 'all'
            The decision rule over the classes, one of 'any', 'all', class name. If 'any', passes if at least one class
            result is above the threshold, if 'all' passes if all of the class results are above the threshold,
            class name, passes if the result for this specified class is above the thershold.
        """

        def condition(check_result, metrics=metrics):
            if metrics is None:
                metrics = list(self.scorers.keys())

            metrics_pass = []

            for metric in metrics:
                if metric not in check_result.Metric.unique():
                    raise DeepchecksValueError(f'The requested metric was not calculated, the metrics calculated in '
                                               f'this check are: {check_result.Metric.unique()}.')

                class_val = check_result[check_result.Metric == metric].groupby('Class Name').Value
                class_gt = class_val.apply(lambda x: x > threshold)
                if class_mode == 'all':
                    metrics_pass.append(all(class_gt))
                elif class_mode == 'any':
                    metrics_pass.append(any(class_gt))
                elif class_mode in class_val.groups:
                    metrics_pass.append(class_gt[class_val.indices[class_mode]].item())
                else:
                    raise DeepchecksValueError(f'class_mode expected be one of the classes in the check results or any '
                                               f'or all, recieved {class_mode}.')

            if all(metrics_pass):
                return ConditionResult(ConditionCategory.PASS, 'Passed for all of the mertics.')
            else:
                failed_metrics = ([a for a, b in zip(metrics, metrics_pass) if not b])
                return ConditionResult(ConditionCategory.FAIL, f'Failed for metrics: {failed_metrics}')

        return self.add_condition(f'Score is greater than {threshold} for classes: {class_mode}', condition)

    def add_condition_less_than(self, threshold: float, metrics: List[str] = None, class_mode: str = 'all'):
        """Add condition - the result is less than the threshold.

        Parameters
        ----------
        threshold: float
            The threshold that the metrics result should be less than.

        metrics: List[str]
            The names of the metrics from the check to apply the condition to. If None, runs on all of the metrics that
            were calculated in the check.

        class_mode: str, default: 'all'
            The decision rule over the classes, one of 'any', 'all', class name. If 'any', passes if at least one class
            result is above the threshold, if 'all' passes if all of the class results are above the threshold,
            class name, passes if the result for this specified class is above the thershold.
        """

        def condition(check_result, metrics=metrics):
            if metrics is None:
                metrics = list(self.scorers.keys())

            metrics_pass = []

            for metric in metrics:
                if metric not in check_result.Metric.unique():
                    raise DeepchecksValueError(f'The requested metric was not calculated, the metrics calculated in '
                                               f'this check are: {check_result.Metric.unique()}.')

                class_val = check_result[check_result.Metric == metric].groupby('Class Name').Value
                class_lt = class_val.apply(lambda x: x < threshold)
                if class_mode == 'all':
                    metrics_pass.append(all(class_lt))
                elif class_mode == 'any':
                    metrics_pass.append(any(class_lt))
                elif class_mode in class_val.groups:
                    metrics_pass.append(class_lt[class_val.indices[class_mode]].item())
                else:
                    raise DeepchecksValueError(f'class_mode expected be one of the classes in the check results or any '
                                               f'or all, recieved {class_mode}.')

            if all(metrics_pass):
                return ConditionResult(ConditionCategory.PASS, 'Passed for all of the mertics.')
            else:
                failed_metrics = ([a for a, b in zip(metrics, metrics_pass) if not b])
                return ConditionResult(ConditionCategory.FAIL, f'Failed for metrics: {failed_metrics}')

        return self.add_condition(f'Score is less than {threshold} for classes: {class_mode}', condition)
