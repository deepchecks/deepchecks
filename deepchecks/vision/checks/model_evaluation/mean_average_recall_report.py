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
"""Module containing mean average recall report check."""
import math
from typing import Tuple, TypeVar

import numpy as np
import pandas as pd

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.condition import ConditionCategory
from deepchecks.utils.strings import format_number
from deepchecks.vision import Batch, Context, SingleDatasetCheck
from deepchecks.vision.metrics_utils.detection_precision_recall import ObjectDetectionAveragePrecision
from deepchecks.vision.vision_data import TaskType

__all__ = ['MeanAverageRecallReport']

MPR = TypeVar('MPR', bound='MeanAverageRecallReport')


class MeanAverageRecallReport(SingleDatasetCheck):
    """Summarize mean average recall metrics on a dataset and model per detections and area range.

    Parameters
    ----------
    area_range: tuple, default: (32**2, 96**2)
        Slices for small/medium/large buckets.
    """

    def __init__(self, area_range: Tuple = (32 ** 2, 96 ** 2), **kwargs):
        super().__init__(**kwargs)
        self._area_range = area_range

    def initialize_run(self, context: Context, dataset_kind: DatasetKind = None):
        """Initialize run by asserting task type and initializing metric."""
        context.assert_task_type(TaskType.OBJECT_DETECTION)
        self._ap_metric = ObjectDetectionAveragePrecision(return_option=None, area_range=self._area_range)

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
        """Update the metrics by passing the batch to ignite metric update method."""
        label = batch.labels
        prediction = batch.predictions
        self._ap_metric.update((prediction, label))

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Compute the metric result using the ignite metrics compute method and create display."""
        small_area = int(math.sqrt(self._area_range[0]))
        large_area = int(math.sqrt(self._area_range[1]))
        res = self._ap_metric.compute()[0]['recall']
        rows = []
        for title, area_name in zip(['All',
                                     f'Small (area < {small_area}^2)',
                                     f'Medium ({small_area}^2 < area < {large_area}^2)',
                                     f'Large (area < {large_area}^2)'],
                                    ['all', 'small', 'medium', 'large']):
            rows.append([
                title,
                self._ap_metric.get_classes_scores_at(res, area=area_name, max_dets=1),
                self._ap_metric.get_classes_scores_at(res, area=area_name, max_dets=10),
                self._ap_metric.get_classes_scores_at(res, area=area_name, max_dets=100)
            ])

        results = pd.DataFrame(data=rows, columns=['Area size', 'AR@1 (%)', 'AR@10 (%)', 'AR@100 (%)'])
        results = results.set_index('Area size')

        return CheckResult(value=results, display=[results])

    def add_condition_test_average_recall_greater_than(self: MPR, min_score: float) -> MPR:
        """Add condition - AR score is greater than given score.

        Parameters
        ----------
        min_score : float
            Minimum score to pass the check.
        """

        def condition(df: pd.DataFrame):
            min_col_per_row = df.idxmin(axis=1)
            min_score_per_row = [df.loc[r, c] for r, c in min_col_per_row.items()]
            loc_min_row = np.argmin(min_score_per_row)
            score = min_score_per_row[loc_min_row]
            area = min_col_per_row.index[loc_min_row]
            iou = min_col_per_row[loc_min_row]
            category = ConditionCategory.FAIL if score < min_score else ConditionCategory.PASS

            details = f'Found lowest score of {format_number(score)} for area {area} and IoU {iou}'
            return ConditionResult(category, details)

        return self.add_condition(f'Scores are greater than {min_score}', condition)
