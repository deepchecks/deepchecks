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
"""Module containing mean average precision report check."""
from collections import defaultdict
import math
from typing import TypeVar, Tuple, Any

import pandas as pd
import plotly.express as px
import numpy as np

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.utils.strings import format_number
from deepchecks.vision import SingleDatasetCheck, Context
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.metrics_utils.detection_precision_recall import AveragePrecision

__all__ = ['MeanAveragePrecisionReport']

MPR = TypeVar('MPR', bound='MeanAveragePrecisionReport')


class MeanAveragePrecisionReport(SingleDatasetCheck):
    """Summarize mean average precision metrics on a dataset and model per IoU and area range.

    Parameters
    ----------
    area_range: tuple, default: (32**2, 96**2)
        Slices for small/medium/large buckets.
    """

    def __init__(self, area_range: Tuple = (32**2, 96**2)):
        super().__init__()
        self._area_range = area_range

    def initialize_run(self, context: Context, dataset_kind: DatasetKind = None):
        """Initialize run by asserting task type and initializing metric."""
        self._ap_metric = AveragePrecision(return_option=None, area_range=self._area_range)
        context.assert_task_type(TaskType.OBJECT_DETECTION)

    def update(self, context: Context, batch: Any, dataset_kind: DatasetKind):
        """Update the metrics by passing the batch to ignite metric update method."""
        dataset = context.get_data_by_kind(dataset_kind)
        label = dataset.label_formatter(batch)
        prediction = context.infer(batch)
        self._ap_metric.update((prediction, label))

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Compute the metric result using the ignite metrics compute method and create display."""
        small_area = int(math.sqrt(self._area_range[0]))
        large_area = int(math.sqrt(self._area_range[1]))
        res = self._ap_metric.compute()[0]['precision']
        rows = []
        for title, area_name in zip(['All',
                                     f'Small (area < {small_area}^2)',
                                     f'Medium ({small_area}^2 < area < {large_area}^2)',
                                     f'Large (area < {large_area}^2)'],
                                    ['all', 'small', 'medium', 'large']):
            area_scores = [title]
            area_scores.append(self._ap_metric.get_classes_scores_at(res, area=area_name, max_dets=100))
            area_scores.append(self._ap_metric.get_classes_scores_at(res, iou=0.5, area=area_name, max_dets=100))
            area_scores.append(self._ap_metric.get_classes_scores_at(res, iou=0.75, area=area_name, max_dets=100))

            rows.append(area_scores)

        results = pd.DataFrame(columns=['Area size', 'mAP@0.5..0.95 (%)', 'AP@.50 (%)', 'AP@.75 (%)'])
        for i in range(len(rows)):
            results.loc[i] = rows[i]
        results = results.set_index('Area size')

        filtered_res = self._ap_metric.filter_res(res, area='all', max_dets=100)
        filtered_res_shape = filtered_res.shape
        filtered_res = np.reshape(filtered_res, (filtered_res_shape[0], filtered_res_shape[3]))
        mean_res = np.zeros(filtered_res_shape[0])
        for i in range(filtered_res_shape[0]):
            mean_res[i] = np.mean(filtered_res[i][filtered_res[i] > -1])

        data = {
            'IoU': self._ap_metric.iou_thresholds,
            'AP (%)': mean_res
        }
        df = pd.DataFrame.from_dict(data)

        fig = px.line(df, x='IoU', y='AP (%)', title='Mean Average Precision over increasing IoU thresholds')

        return CheckResult(value=results, display=[results, fig])

    def add_condition_test_average_precision_not_less_than(self: MPR, min_score: float) -> MPR:
        """Add condition - mAP score is not less than given score.

        Parameters
        ----------
        min_score : float
            Minimum score to pass the check.
        """
        def condition(df: pd.DataFrame):
            not_passed = defaultdict(dict)
            for index, column in zip(df.index, df.columns):
                if df.loc[index, column] < min_score:
                    not_passed[index][column] = format_number(df.loc[index, column], 3)
            if len(not_passed):
                details = f'Found scores below threshold:\n' \
                          f'{dict(not_passed)}'
                return ConditionResult(False, details)
            return ConditionResult(True)

        return self.add_condition(f'Scores are not less than {min_score}', condition)
