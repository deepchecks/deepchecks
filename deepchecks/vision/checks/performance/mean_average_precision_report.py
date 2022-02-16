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
"""Module containing class performance check."""
from typing import TypeVar, Any

import pandas as pd
import plotly.express as px
import numpy as np

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.vision import SingleDatasetCheck, Context
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.metrics_utils.detection_precision_recall import AveragePrecision

__all__ = ['MeanAveragePrecisionReport']

MPR = TypeVar('MPR', bound='MeanAveragePrecisionReport')


class MeanAveragePrecisionReport(SingleDatasetCheck):
    """Summarize given metrics on a dataset and model."""

    def initialize_run(self, context: Context, dataset_kind: DatasetKind = None):
        """Initialize run by asserting task type and initializing metric."""
        context.assert_task_type(TaskType.OBJECT_DETECTION)
        self._ap_metric = AveragePrecision(return_option=None)

    def update(self, context: Context, batch: Any, dataset_kind: DatasetKind):
        """Update the metrics by passing the batch to ignite metric update method."""
        dataset = context.get_data_by_kind(dataset_kind)
        images = batch[0]
        label = dataset.label_transformer(batch[1])
        prediction = context.prediction_formatter(context.infer(images))
        self._ap_metric.update((prediction, label))

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Compute the metric result using the ignite metrics compute method and create display."""
        res = self._ap_metric.compute()[0]['precision']
        rows = []
        for title, area_name in zip(['All', 'Small (area<32^2)', 'Medium (32^2<area<96^2)', 'Large (area<96^2)'],
                                ['all', 'small', 'medium', 'large']):
            area_scores = [title]
            area_scores.append(self._ap_metric.get_classes_scores_at(res, area=area_name, max_dets=100))
            area_scores.append(self._ap_metric.get_classes_scores_at(res, iou=0.5, area=area_name, max_dets=100))
            area_scores.append(self._ap_metric.get_classes_scores_at(res, iou=0.75, area=area_name, max_dets=100))

            rows.append(area_scores)

        results = pd.DataFrame(columns=['Area size', 'mAP@.5...95 (%)', 'AP@.50 (%)', 'AP@.75 (%)'])
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

        fig = px.line(df, x='IoU', y='AP (%)', title='Mean average precision over increasing IoU thresholds')

        return CheckResult(value=results, display=[results, fig])

    def add_condition_test_average_precision_not_less_than(self: MPR, min_score: float)-> MPR:
        """Add condition - mAP score is not less than given score.

        Parameters
        ----------
        min_score : float
            Minimum score to pass the check.
        """
        def condition(df: pd.DataFrame):
            not_passed = []
            for column in df.columns:
                not_passed_in_col = df.loc[df[column] < min_score].loc[:,column]
                if len(not_passed_in_col):
                    not_passed.append(not_passed_in_col)
            if len(not_passed):
                details = f'Found scores below threshold:\n' \
                          f'{pd.DataFrame(not_passed).T.to_dict("records")}'
                return ConditionResult(False, details)
            return ConditionResult(True)

        return self.add_condition(f'Scores are not less than {min_score}', condition)
