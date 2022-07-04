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
import numbers
import typing as t
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

__all__ = ['SingleDatasetScalarPerformance']


class SingleDatasetScalarPerformance(SingleDatasetCheck):
    """Calculate a performance metric as a scalar for a given model and a given dataset.

    Parameters
    ----------
    metric: Metric, default: None
        An ignite.Metric object whose score should be used. If None is given, use the default metric.
    reduce: torch function, default: None
        The function to reduce the scores tensor into a single scalar. For metrics that return a scalar use None
        (default).
    metric_name: str, default: None
        A name for the metric to show in the check results.
    reduce_name: str, default: None
        A name for the reduce function to show in the check results.

    """

    def __init__(self,
                 metric: Metric = None,
                 reduce: t.Callable = None,
                 metric_name: str = None,
                 reduce_name: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.metric = metric
        self.reduce = reduce

        self.metric_name = metric_name or (metric.__class__.__name__ if metric else None)
        self.reduce_name = reduce_name or (reduce.__name__ if reduce else None)

    def initialize_run(self, context: Context, dataset_kind: DatasetKind.TRAIN):
        """Initialize the metric for the check, and validate task type is relevant."""
        if self.metric is None:
            if context.train.task_type == TaskType.CLASSIFICATION:
                self.metric = Accuracy()
                if self.metric_name is None:
                    self.metric_name = 'accuracy'
            elif context.train.task_type == TaskType.OBJECT_DETECTION:
                self.metric = ObjectDetectionAveragePrecision()
                if self.metric_name is None:
                    self.metric_name = 'object_detection_average_precision'
                if self.reduce is None:
                    self.reduce = torch.nanmean
                    self.reduce_name = 'nan_mean'
            else:
                raise DeepchecksValueError('For task types other then classification or object detection, '
                                           'pass a metric explicitly')
        self.metric.reset()

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind.TRAIN):
        """Update the metrics by passing the batch to ignite metric update method."""
        label = batch.labels
        prediction = batch.predictions
        self.metric.update((prediction, label))

    def compute(self, context: Context, dataset_kind: DatasetKind.TRAIN) -> CheckResult:
        """Compute the metric result using the ignite metrics compute method and reduce to a scalar."""
        metric_result = self.metric.compute()
        if self.reduce is not None:
            if isinstance(metric_result, numbers.Real):
                warnings.warn(SyntaxWarning('Metric result is already scalar, skipping reduce operation.'
                                            'Pass reduce=None to prevent this'))
                result_value = float(metric_result)
            else:
                result_value = float(self.reduce(metric_result))
        elif isinstance(metric_result, float):
            result_value = metric_result
        else:
            raise DeepchecksValueError(f'The metric {self.metric.__class__} return a non-scalar value, '
                                       f'please specify a reduce function or choose a different metric')

        result_dict = {'score': result_value,
                       'metric': self.metric_name,
                       'reduce': self.reduce_name}
        return CheckResult(result_dict)

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
