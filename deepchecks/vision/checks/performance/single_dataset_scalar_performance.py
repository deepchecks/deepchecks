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
import typing as t
import torch
from ignite.metrics import Metric, Accuracy
from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.condition import ConditionCategory
from deepchecks.vision import Batch, Context, SingleDatasetCheck
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.metrics_utils.object_detection_precision_recall import \
    ObjectDetectionAveragePrecision
import warnings

__all__ = ['SingleDatasetScalarPerformance']


class SingleDatasetScalarPerformance(SingleDatasetCheck):
    """Calculate a performance metric as a scalar for a given model and a given dataset.

    Parameters
    ----------
        metric: Metric,  default: None
        An ignite.Metric object whose score should be used. If None is given, use the default metric.
        reduce: torch function, default: None
        The function to reduce the scores tensor into a single scalar. For metrics that return a scalar use None
        (default).
    """
    def __init__(self,
                 metric: Metric = None,
                 reduce: t.Callable = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.metric = metric
        self.reduce = reduce

    def initialize_run(self, context: Context, dataset_kind: DatasetKind.TRAIN):
        """Initialize the metric for the check, and validate task type is relevant."""
        if self.metric is None:
            if context.train.task_type == TaskType.CLASSIFICATION:
                self.metric = Accuracy()
            elif context.train.task_type == TaskType.OBJECT_DETECTION:
                self.metric = ObjectDetectionAveragePrecision()
                if self.reduce is None:
                    self.reduce = torch.nanmean
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
            if type(metric_result) is float:
                warnings.warn(SyntaxWarning('Metric result is already scalar, skipping reduce operation'))
                result_value = metric_result
            else:
                result_value = float(self.reduce(metric_result))
        elif type(metric_result) is float:
            result_value = metric_result
        else:
            raise DeepchecksValueError(f'The metric {self.metric.__class__} return a non-scalar value, '
                                       f'please specify a reduce function or choose a different metric')
        result_dict = {'score': result_value,
                       'metric': self.metric.__class__.__name__,
                       'reduce': None if self.reduce is None else self.reduce.__name__}
        return CheckResult(result_dict)

    def add_condition_greater_than(self, threshold: float) -> ConditionResult:
        """Add condition - the result is greater than the threshold"""
        def condition(check_result):
            if check_result['score'] > threshold:
                return ConditionResult(ConditionCategory.PASS)
            else:
                details = f'The score is not greater than {threshold}'
                return ConditionResult(ConditionCategory.FAIL, details)
        return self.add_condition(f'Score is greater than {threshold}', condition)

    def add_condition_greater_equal_to(self, threshold: float) -> ConditionResult:
        """Add condition - the result is greater than the threshold"""

        def condition(check_result):
            if check_result['score'] >= threshold:
                return ConditionResult(ConditionCategory.PASS)
            else:
                details = f'The score is not greater than or equal to {threshold}'
                return ConditionResult(ConditionCategory.FAIL, details)

        return self.add_condition(f'Score is greater than or equal to {threshold}', condition)

    def add_condition_less_than(self, threshold: float) -> ConditionResult:
        """Add condition - the result is greater than the threshold"""
        def condition(check_result):
            if check_result['score'] < threshold:
                return ConditionResult(ConditionCategory.PASS)
            else:
                details = f'The score is not less than {threshold}'
                return ConditionResult(ConditionCategory.FAIL, details)
        return self.add_condition(f'Score is less than {threshold}', condition)

    def add_condition_less_equal_to(self, threshold: float) -> ConditionResult:
        """Add condition - the result is greater than the threshold"""

        def condition(check_result):
            if check_result['score'] <= threshold:
                return ConditionResult(ConditionCategory.PASS)
            else:
                details = f'The score is not less than or equal to {threshold}'
                return ConditionResult(ConditionCategory.FAIL, details)

        return self.add_condition(f'Score is less than or equal to {threshold}', condition)