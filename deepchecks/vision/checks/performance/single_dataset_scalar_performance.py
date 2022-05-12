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
"""Module containing a single dataset with a signle scalar output performance check."""
import typing as t
import torch
from ignite.metrics import Metric
from deepchecks.core import CheckResult, DatasetKind
from deepchecks.vision import Batch, Context, SingleDatasetCheck
from deepchecks.vision.metrics_utils import (get_scorers_list)

__all__ = ['SingleDatasetScalarPerformance']


class SingleDatasetScalarPerformance(SingleDatasetCheck):
    """Summarize a given metric over a given dataset to return a performance score as a scalar.

    Parameters
    ----------
        metric : default: None
        An ignite.Metric object whose score should be used. If None are given, use the default metrics.
        reduce: torch function, default: torch.mean
        The function to reduce the scores vector into a single scalar
    """
    def __init__(self,
                 metric: Metric,
                 reduce: t.Callable = torch.mean,
                 **kwargs):
        super().__init__(**kwargs)
        self.metric = metric
        self.reduce = reduce

    def initialize_run(self, context: Context, dataset_kind: DatasetKind.TRAIN):
        """Initialize the metric for the check, and validate task type is relevant."""
        self.metric.reset()

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind.TRAIN):
        """Update the metrics by passing the batch to ignite metric update method."""
        label = batch.labels
        prediction = batch.predictions
        self.metric.update((prediction, label))

    def compute(self, context: Context, dataset_kind: DatasetKind.TRAIN) -> CheckResult:
        """Compute the metric result using the ignite metrics compute method and reduce to a scalar."""
        result = self.reduce(self.metric.compute())
        return CheckResult(result)
