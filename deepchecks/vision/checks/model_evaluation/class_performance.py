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
from typing import Dict, List, TypeVar, Union

import pandas as pd
import plotly.express as px
from ignite.metrics import Metric

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.check_utils.class_performance_utils import (
    get_condition_class_performance_imbalance_ratio_less_than, get_condition_test_performance_greater_than,
    get_condition_train_test_relative_degradation_less_than)
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils import plot
from deepchecks.utils.strings import format_percent
from deepchecks.vision import Batch, Context, TrainTestCheck
from deepchecks.vision.metrics_utils.scorers import filter_classes_for_display, get_scorers_dict, metric_results_to_df

__all__ = ['ClassPerformance']


PR = TypeVar('PR', bound='ClassPerformance')


class ClassPerformance(TrainTestCheck):
    """Summarize given metrics on a dataset and model.

    Parameters
    ----------
    alternative_metrics : Union[Dict[str, Union[Metric,str]], List[str]], default: None
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

    def __init__(self,
                 alternative_metrics: Union[Dict[str, Union[Metric, str]], List[str]] = None,
                 n_to_show: int = 20,
                 show_only: str = 'largest',
                 metric_to_show_by: str = None,
                 class_list_to_show: List[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
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
        self._data_metrics = {}

    def initialize_run(self, context: Context):
        """Initialize run by creating the _state member with metrics for train and test."""
        self._data_metrics = {}
        self._data_metrics[DatasetKind.TRAIN] = get_scorers_dict(context.train,
                                                                 alternative_scorers=self.alternative_metrics)
        self._data_metrics[DatasetKind.TEST] = get_scorers_dict(context.train,
                                                                alternative_scorers=self.alternative_metrics)

        if not self.metric_to_show_by:
            self.metric_to_show_by = list(self._data_metrics[DatasetKind.TRAIN].keys())[0]

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
        """Update the metrics by passing the batch to ignite metric update method."""
        label = batch.labels
        prediction = batch.predictions
        for _, metric in self._data_metrics[dataset_kind].items():
            metric.update((prediction, label))

    def compute(self, context: Context) -> CheckResult:
        """Compute the metric result using the ignite metrics compute method and create display."""
        results = []
        for dataset_kind in [DatasetKind.TRAIN, DatasetKind.TEST]:
            dataset = context.get_data_by_kind(dataset_kind)
            metrics_df = metric_results_to_df(
                {k: m.compute() for k, m in self._data_metrics[dataset_kind].items()}, dataset
            )
            metrics_df['Dataset'] = dataset_kind.value
            metrics_df['Number of samples'] = metrics_df['Class'].map(dataset.n_of_samples_per_class.get)
            results.append(metrics_df)

        results_df = pd.concat(results)
        results_df = results_df[['Dataset', 'Metric', 'Class', 'Class Name', 'Number of samples', 'Value']]
        results_df = results_df.sort_values(by=['Dataset', 'Value'], ascending=False)

        if context.with_display:
            if self.class_list_to_show is not None:
                display_df = results_df.loc[results_df['Class'].isin(self.class_list_to_show)]
            elif self.n_to_show is not None:
                rows = results_df['Class'].isin(filter_classes_for_display(
                    results_df,
                    self.metric_to_show_by,
                    self.n_to_show,
                    self.show_only
                ))
                display_df = results_df.loc[rows]
            else:
                display_df = results_df

            fig = (
                px.histogram(
                    display_df,
                    x='Class Name',
                    y='Value',
                    color='Dataset',
                    color_discrete_sequence=(plot.colors['Train'], plot.colors['Test']),
                    barmode='group',
                    facet_col='Metric',
                    facet_col_spacing=0.05,
                    hover_data=['Number of samples'])
                .update_xaxes(title='Class', type='category')
                .update_yaxes(title='Value', matches=None)
                .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
                .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
            )
        else:
            fig = None

        return CheckResult(
            results_df,
            header='Class Performance',
            display=fig
        )

    def add_condition_test_performance_greater_than(self: PR, min_score: float) -> PR:
        """Add condition - metric scores are greater than the threshold.

        Parameters
        ----------
        min_score : float
            Minimum score to pass the check.
        """
        condition = get_condition_test_performance_greater_than(min_score=min_score)

        return self.add_condition(f'Scores are greater than {min_score}', condition)

    def add_condition_train_test_relative_degradation_less_than(self: PR, threshold: float = 0.1) -> PR:
        """Add condition - test performance is not degraded by more than given percentage in train.

        Parameters
        ----------
        threshold : float , default: 0.1
            maximum degradation ratio allowed (value between 0 and 1)
        """
        condition = get_condition_train_test_relative_degradation_less_than(threshold=threshold)

        return self.add_condition(f'Train-Test scores relative degradation is less than {threshold}',
                                  condition)

    def add_condition_class_performance_imbalance_ratio_less_than(
        self: PR,
        threshold: float = 0.3,
        score: str = None
    ) -> PR:
        """Add condition - relative ratio difference between highest-class and lowest-class is less than threshold.

        Parameters
        ----------
        threshold : float , default: 0.3
            ratio difference threshold
        score : str , default: None
            limit score for condition

        Returns
        -------
        Self
            instance of 'ClassPerformance' or it subtype

        Raises
        ------
        DeepchecksValueError
            if unknown score function name were passed.
        """
        if score is None:
            raise DeepchecksValueError('Must define "score" parameter')

        condition = get_condition_class_performance_imbalance_ratio_less_than(threshold=threshold, score=score)

        return self.add_condition(
            name=f'Relative ratio difference between labels \'{score}\' score is less than {format_percent(threshold)}',
            condition_func=condition
        )
