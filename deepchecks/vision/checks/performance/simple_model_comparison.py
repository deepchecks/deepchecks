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

import numpy as np
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

from deepchecks.vision.metrics_utils.metrics import filter_classes_for_display

_allowed_strategies = (
    'most_frequent',
    'prior',
    'stratified',
    'uniform'
)


class SimpleModelComparison(TrainTestCheck):
    """Compare given model score to simple model score (according to given model type).

    For classification models, the simple model is a dummy classifier the selects the predictions based on a strategy.


    Parameters
    ----------
    strategy : str, default='prior'
        Strategy to use to generate the predictions of the simple model.

        * 'most_frequent' : The most frequent label in the training set is predicted.
          The probability vector is 1 for the most frequent label and 0 for the other predictions.
        * 'prior' : The probability vector always contains the empirical class prior distribution (i.e. the class
          distribution observed in the training set).
        * 'stratified' : The predictions are generated by sampling one-hot vectors from a multinomial distribution
          parametrized by the empirical class prior probabilities.
        * 'uniform' : Generates predictions uniformly at random from the list of unique classes observed in y,
          i.e. each class has equal probability. The predicted class is chosen randomly.
    alternative_metrics : Dict[str, Metric], default: None
        A dictionary of metrics, where the key is the metric name and the value is an ignite.Metric object whose score
        should be used. If None are given, use the default metrics.
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
                 strategy: str = 'most_frequent',
                 alternative_metrics: Dict[str, Metric] = None,
                 n_to_show: int = 20,
                 show_only: str = 'largest',
                 metric_to_show_by: str = None,
                 class_list_to_show: List[int] = None
                 ):
        super().__init__()
        self.strategy = strategy

        if self.strategy not in _allowed_strategies:
            raise DeepchecksValueError(
                f'Unknown strategy type: {self.strategy}, expected one of{_allowed_strategies}.'
            )

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
        self._state = {}

    def initialize_run(self, context: Context):
        """Initialize the metrics for the check, and validate task type is relevant."""
        context.assert_task_type(TaskType.CLASSIFICATION)

        self._state[DatasetKind.TEST.value] = get_scorers_list(context.train, self.alternative_metrics)
        self._state['Simple Model'] = get_scorers_list(context.train, self.alternative_metrics)

        if context.train.task_type == TaskType.CLASSIFICATION:
            class_prior = np.zeros(context.train.n_of_classes)
            n_samples = 0
            for label, total in context.train.n_of_samples_per_class.items():
                class_prior[label] = total
                n_samples += total
            class_prior /= n_samples

            if self.strategy == 'most_frequent':
                dummy_prediction = np.zeros(context.train.n_of_classes)
                dummy_prediction[np.argmax(class_prior)] = 1
                self._state['dummy_prediction_generator'] = lambda: torch.from_numpy(dummy_prediction)
            elif self.strategy == 'prior':
                self._state['dummy_prediction_generator'] = lambda: torch.from_numpy(class_prior)
            elif self.strategy == 'stratified':
                self._state['dummy_prediction_generator'] = \
                    lambda: torch.from_numpy(np.random.multinomial(1, class_prior))
            elif self.strategy == 'uniform':
                self._state['dummy_prediction_generator'] = \
                    lambda: torch.from_numpy(np.ones(context.train.n_of_classes) / context.train.n_of_classes)
            else:
                raise DeepchecksValueError(
                    f'Unknown strategy type: {self.strategy}, expected one of {_allowed_strategies}.'
                )

        if not self.metric_to_show_by:
            self.metric_to_show_by = list(self._state[DatasetKind.TEST.value].keys())[0]

    def update(self, context: Context, batch: Any, dataset_kind: DatasetKind):
        """Update the metrics for the check."""
        if dataset_kind == DatasetKind.TEST:
            dataset = context.get_data_by_kind(dataset_kind)
            label = dataset.label_formatter(batch)
            prediction = context.infer(batch)
            for _, metric in self._state[DatasetKind.TEST.value].items():
                metric.update((prediction, label))
            for _, metric in self._state['Simple Model'].items():
                pred = []
                for _ in range(len(label)):
                    pred.append(self._state['dummy_prediction_generator']())
                metric.update((torch.stack(pred), label))

    def compute(self, context: Context) -> CheckResult:
        """Compute the metrics for the check."""
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
            classes_to_show = filter_classes_for_display(results_df,
                                                         self.metric_to_show_by,
                                                         self.n_to_show,
                                                         self.show_only)
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
