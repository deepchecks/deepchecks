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
import math
from collections import defaultdict
from typing import Tuple, TypeVar

import numpy as np
import pandas as pd
import plotly.express as px

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.condition import ConditionCategory
from deepchecks.utils.strings import format_number
from deepchecks.vision import Batch, Context, SingleDatasetCheck
from deepchecks.vision.metrics_utils.object_detection_precision_recall import ObjectDetectionAveragePrecision
from deepchecks.vision.vision_data import TaskType

__all__ = ['MeanAveragePrecisionReport']

MPR = TypeVar('MPR', bound='MeanAveragePrecisionReport')


class MeanAveragePrecisionReport(SingleDatasetCheck):
    """Summarize mean average precision metrics on a dataset and model per IoU and bounding box area.

    Parameters
    ----------
    area_range: tuple, default: (32**2, 96**2)
        Slices for small/medium/large buckets.
    """

    def __init__(self, area_range: Tuple = (32**2, 96**2), **kwargs):
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

        results = pd.DataFrame(columns=['Area size', 'mAP@[.50::.95] (avg.%)', 'mAP@.50 (%)', 'mAP@.75 (%)'])
        for i in range(len(rows)):
            results.loc[i] = rows[i]
        results = results.set_index('Area size')

        filtered_res = self._ap_metric.filter_res(res, area='all', max_dets=100)
        filtered_res_shape = filtered_res.shape
        filtered_res = np.reshape(filtered_res, (filtered_res_shape[0], filtered_res_shape[3]))
        mean_res = np.zeros(filtered_res_shape[0])
        for i in range(filtered_res_shape[0]):
            mean_res[i] = np.nanmean(filtered_res[i][filtered_res[i] > -1])

        data = {
            'IoU threshold': self._ap_metric.iou_thresholds,
            'mAP (%)': mean_res
        }
        df = pd.DataFrame.from_dict(data)

        fig = px.line(df, x='IoU threshold', y='mAP (%)',
                      title='Mean Average Precision over increasing IoU thresholds')

        return CheckResult(value=results, display=[results, fig])

    def add_condition_mean_average_precision_not_less_than(self: MPR, min_score: float) -> MPR:
        """Add condition - mAP scores for in different area thresholds is not less than given score.

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
                return ConditionResult(ConditionCategory.FAIL, details)
            return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(f'Scores are not less than {min_score}', condition)

    def add_condition_average_mean_average_precision_not_less_than(self: MPR, min_score: float = 0.3) -> MPR:
        """Add condition - average mAP for IoU values between 0.5 to 0.9 in all areas is not less than given score.

        Parameters
        ----------
        min_score : float
            Minimum score to pass the check.
        """
        def condition(df: pd.DataFrame):
            df = df.reset_index()
            value = df.loc[df['Area size'] == 'All', :]['mAP@[.50::.95] (avg.%)'][0]
            if value < min_score:
                details = f'mAP score is: {format_number(value)}'
                return ConditionResult(ConditionCategory.FAIL, details)
            return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(f'mAP score is not less than {min_score}', condition)
