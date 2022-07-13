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
from typing import Callable, Dict, Union
import warnings

import torch
from ignite.metrics import Accuracy, Metric

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.strings import format_number
from deepchecks.vision import Batch, Context, SingleDatasetCheck
from deepchecks.vision.metrics_utils.detection_precision_recall import ObjectDetectionAveragePrecision
from deepchecks.vision.vision_data import TaskType
from deepchecks.core.checks import ReduceMixin
from deepchecks.vision.metrics_utils.metrics import get_scorers_dict, calculate_metrics, metric_results_to_df

__all__ = ['SingleDatasetScalarPerformance']


class SingleDatasetScalarPerformance(SingleDatasetCheck, ReduceMixin):
    """Calculate a performance metric as a scalar for a given model and a given dataset.

    Parameters
    ----------
    alternative_scorers : Dict[str, Callable], default: None
        An optional dictionary of scorer name to scorer functions.
        If none given, using default scorers
    reduce: Union[Callable, str], default: 'mean'
        An optional argument only used for the reduce_output function when using
        non-average scorers.

    """

    def __init__(self,
                 alternative_scorers : Dict[str, Union[Callable, Metric]] = None,
                 reduce: Union[Callable, str] = 'mean',
                 **kwargs):
        super().__init__(**kwargs)
        self.user_scorers = alternative_scorers
        self.reduce = reduce


    def initialize_run(self, context: Context, dataset_kind: DatasetKind.TRAIN):
        """Initialize the metric for the check, and validate task type is relevant."""
        self.scorers = get_scorers_dict(context.get_data_by_kind(dataset_kind), self.user_scorers)

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
        return CheckResult(result_df)

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return the values of the metrics for the test dataset in {metric: value} format."""
        reduced_output = check_result.value.aggeregate(self.reduce).to_dict()
        return reduced_output


    def add_condition_greater_than(self, threshold: float) -> ConditionResult:
        """Add condition - the result is greater than the threshold."""
        def condition(check_result):
            details = f'The score {self.metric_name} is {format_number(check_result["score"])}'
            if check_result['score'] > threshold:
                return ConditionResult(ConditionCategory.PASS, details)
            else:
                return ConditionResult(ConditionCategory.FAIL, details)
        return self.add_condition(f'Score is greater than {threshold}', condition)

    def add_condition_greater_or_equal(self, threshold: float) -> ConditionResult:
        """Add condition - the result is greater or equal to the threshold."""
        def condition(check_result):
            details = f'The score {self.metric_name} is {format_number(check_result["score"])}'
            if check_result['score'] >= threshold:
                return ConditionResult(ConditionCategory.PASS, details)
            else:
                return ConditionResult(ConditionCategory.FAIL, details)
        return self.add_condition(f'Score is greater or equal to {threshold}', condition)

    def add_condition_less_than(self, threshold: float) -> ConditionResult:
        """Add condition - the result is less than the threshold."""
        def condition(check_result):
            details = f'The score {self.metric_name} is {format_number(check_result["score"])}'
            if check_result['score'] < threshold:
                return ConditionResult(ConditionCategory.PASS, details)
            else:
                return ConditionResult(ConditionCategory.FAIL, details)
        return self.add_condition(f'Score is less than {threshold}', condition)

    def add_condition_less_or_equal(self, threshold: float) -> ConditionResult:
        """Add condition - the result is less or equal to the threshold."""

        def condition(check_result):
            details = f'The score {self.metric_name} is {format_number(check_result["score"])}'
            if check_result['score'] <= threshold:
                return ConditionResult(ConditionCategory.PASS, details)
            else:
                return ConditionResult(ConditionCategory.FAIL, details)

        return self.add_condition(f'Score is less or equal to {threshold}', condition)
