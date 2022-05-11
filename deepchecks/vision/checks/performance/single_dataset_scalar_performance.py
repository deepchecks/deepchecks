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
import math
import typing as t
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from ignite.metrics import Metric

from deepchecks import ConditionResult
from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.condition import ConditionCategory
from deepchecks.utils import plot
from deepchecks.utils.strings import format_number, format_percent
from deepchecks.vision import Batch, Context, SingleDatasetCheck
from deepchecks.vision.metrics_utils import (get_scorers_list,
                                             metric_results_to_df)
from deepchecks.vision.utils.image_properties import (default_image_properties,
                                                      validate_properties)


__all__ = ['SingleDatasetScalarPerformance']

class SingleDatasetScalarPerformance(SingleDatasetCheck):
    """Summarize a given metric over a given dataset to return a performance score as a scalar.

    Parameters
    ----------
        metric : default: None
        An ignite.Metric object whose score should be used. If None are given, use the default metrics.
        reduce: str, default: 'mean'
        The function to reduce the scores vector into a single scalar
    """
    def __init__(self,
                 metric: Metric,
                 reduce: str = 'mean'):
        super().__init__(**kwargs)
        self.metric = metric
        self.reduce = reduce

    def initialize_run(self, context: Context):
        """Initialize run by creating the _state member with a metric."""
        pass

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
        """Update the metrics by passing the batch to ignite metric update method."""
        label = batch.labels
        prediction = batch.predictions
        self.metric.update((prediction, label))

    def compute(self, context: Context) -> CheckResult:
        """Compute the metric result using the ignite metrics compute method."""
        pass
