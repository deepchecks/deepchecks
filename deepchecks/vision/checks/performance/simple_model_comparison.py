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
"""Module containing simple comparison check."""
from typing import Dict, Hashable, List, Any

import pandas as pd
import plotly.express as px
import torch
from ignite.metrics import Metric

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import Context, TrainTestCheck
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.metrics_utils import get_scorers_list, metric_results_to_df

__all__ = ['SimpleModelComparison']


class SimpleModelComparison(TrainTestCheck):
    """Compare given model score to simple model score (according to given model type).

    For classification models, the simple model is a dummy classifier that predicts the most common prediction.


    Parameters
    ----------
    alternative_metrics : List[Metric], default: None
            A list of ignite.Metric objects whose score should be used. If None are given, use the default metrics.
    n_to_show : int, default: 20
        Number of classes to show in the report. If None, show all classes.
    show_only : str, default: 'largest'
        Specify which classes to show in the report. Can be one of the following:
        - 'largest': Show the largest classes.
        - 'smallest': Show the smallest classes.
        - 'random': Show random classes.
        - 'best': Show the classes with the highest score.
        - 'worst': Show the classes with the lowest score.
    metric_to_show_by : str, default: None
        Specify the metric to sort the results by. Relevant only when show_only is 'best' or 'worst'.
        If None, sorting by the first metric in the default metrics list.
    class_list_to_show: List[int], default: None
        Specify the list of classes to show in the report. If specified, n_to_show, show_only and metric_to_show_by
        are ignored.

    """

    _state: Dict[Hashable, Any] = {}

    def __init__(self,
                 alternative_metrics: List[Metric] = None,
                 n_to_show: int = 20,
                 show_only: str = 'largest',
                 metric_to_show_by: str = None,
                 class_list_to_show: List[int] = None
                 ):
        super().__init__()
        self.alternative_metrics = alternative_metrics

        self.n_to_show = n_to_show
        self.class_list_to_show = class_list_to_show

        if self.class_list_to_show is None:
            if show_only not in ['largest', 'smallest', 'random', 'best', 'worst']:
                raise DeepchecksValueError(f'Invalid value for show_only: {show_only}. Should be one of: '
                                           f'["largest", "smallest", "random", "best", "worst"]')

            self.show_only = show_only
            if alternative_metrics is not None and show_only in ['best', 'worst'] and metric_to_show_by is None:
                raise DeepchecksValueError('When alternative_metrics are provided and show_only is one of: '
                                           '["best", "worst"], metric_to_show_by must be specified.')

        self.metric_to_show_by = metric_to_show_by
        self._state = dict()

    def initialize_run(self, context: Context):
        """Initialize the metrics for the check, and validate task type is relevant."""
        context.assert_task_type(TaskType.CLASSIFICATION)

        self._state[DatasetKind.TEST.value] = get_scorers_list(context.train, self.alternative_metrics)
        self._state['Simple Model'] = get_scorers_list(context.train, self.alternative_metrics)

        if context.train.task_type == TaskType.CLASSIFICATION:
            predicted_class = max(context.train.n_of_samples_per_class.items(), key=lambda x : x[1])[0]
            dummy_prediction = [0] * context.train.n_of_classes
            dummy_prediction[predicted_class] = 1
            self._state['dummy_prediction'] = torch.Tensor(dummy_prediction)

        if not self.metric_to_show_by:
            self.metric_to_show_by = list(self._state[DatasetKind.TEST.value].keys())[0]

    def update(self, context: Context, batch: Any, dataset_kind: DatasetKind):
        if dataset_kind == DatasetKind.TEST:
            dataset = context.get_data_by_kind(dataset_kind)
            label = dataset.label_transformer(batch)
            prediction = context.infer(batch)
            for _, metric in self._state[DatasetKind.TEST.value].items():
                metric.update((prediction, label))
            for _, metric in self._state['Simple Model'].items():
                metric.update((torch.stack([self._state['dummy_prediction']] * len(label)), label))

    def compute(self, context: Context) -> CheckResult:
        results = []
        for eval_kind in [DatasetKind.TEST.value, 'Simple Model']:
            dataset = context.get_data_by_kind(DatasetKind.TEST)
            metrics_df = metric_results_to_df(
                {k: m.compute() for k, m in self._state[eval_kind].items()}, dataset
            )
            metrics_df['Dataset'] = eval_kind
            metrics_df['Number of samples'] = metrics_df['Class'].map(dataset.n_of_samples_per_class.get)
            results.append(metrics_df)

        results_df = pd.concat(results)

        if self.class_list_to_show is not None:
            results_df = results_df.loc[results_df['Class'].isin(self.class_list_to_show)]
        elif self.n_to_show is not None:
            classes_to_show = self._filter_classes(results_df)
            results_df = results_df.loc[results_df['Class'].isin(classes_to_show)]

        results_df = results_df.sort_values(by=['Dataset', 'Value'], ascending=False)

        fig = px.histogram(
            results_df,
            x='Class Name',
            y='Value',
            color='Dataset',
            barmode='group',
            facet_col='Metric',
            facet_col_spacing=0.05,
            hover_data=['Number of samples']
        )

        if context.train.task_type == TaskType.CLASSIFICATION:
            fig.update_xaxes(tickprefix='Class ', tickangle=60)

        fig = (
            fig.update_xaxes(title=None, type='category')
                .update_yaxes(title=None, matches=None)
                .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
                .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
        )

        return CheckResult(
            results_df,
            header='Simple Model Comparison',
            display=fig
        )

    def _filter_classes(self, metrics_df: pd.DataFrame) -> list:
        # working only on the test set
        tests_metrics_df = metrics_df[(metrics_df['Dataset'] == DatasetKind.TEST.value) &
                                      (metrics_df['Metric'] == self.metric_to_show_by)]
        if self.show_only == 'largest':
            tests_metrics_df = tests_metrics_df.sort_values(by='Number of samples', ascending=False)
        elif self.show_only == 'smallest':
            tests_metrics_df = tests_metrics_df.sort_values(by='Number of samples', ascending=True)
        elif self.show_only == 'random':
            tests_metrics_df = tests_metrics_df.sample(frac=1)
        elif self.show_only == 'best':
            tests_metrics_df = tests_metrics_df.sort_values(by='Value', ascending=False)
        elif self.show_only == 'worst':
            tests_metrics_df = tests_metrics_df.sort_values(by='Value', ascending=True)
        else:
            raise ValueError(f'Unknown show_only value: {self.show_only}')

        return tests_metrics_df.head(self.n_to_show)['Class'].to_list()