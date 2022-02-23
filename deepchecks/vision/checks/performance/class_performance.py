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
from typing import TypeVar, List, Any, Dict

import pandas as pd
import plotly.express as px
from ignite.metrics import Metric

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.strings import format_percent, format_number
from deepchecks.vision import TrainTestCheck, Context
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.metrics_utils.metrics import get_scorers_list, metric_results_to_df, \
    get_default_classification_scorers, get_default_object_detection_scorers, filter_classes_for_display

__all__ = ['ClassPerformance']

PR = TypeVar('PR', bound='ClassPerformance')


class ClassPerformance(TrainTestCheck):
    """Summarize given metrics on a dataset and model.

    Parameters
    ----------
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

    def __init__(self,
                 alternative_metrics: Dict[str, Metric] = None,
                 n_to_show: int = 20,
                 show_only: str = 'largest',
                 metric_to_show_by: str = None,
                 class_list_to_show: List[int] = None):
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

        self._state = {}

    def initialize_run(self, context: Context):
        """Initialize run by creating the _state member with metrics for train and test."""
        context.assert_task_type(TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION)

        if not self.metric_to_show_by:
            if context.train.task_type == TaskType.CLASSIFICATION:
                self.metric_to_show_by = list(get_default_classification_scorers().keys())[0]
            elif context.train.task_type == TaskType.OBJECT_DETECTION:
                self.metric_to_show_by = list(get_default_object_detection_scorers().keys())[0]
            else:
                raise DeepchecksValueError(f'Invalid task type: {context.train.task_type}')

        self._state = {DatasetKind.TRAIN: {}, DatasetKind.TEST: {}}
        self._state[DatasetKind.TRAIN]['scorers'] = get_scorers_list(context.train, self.alternative_metrics)
        self._state[DatasetKind.TEST]['scorers'] = get_scorers_list(context.train, self.alternative_metrics)

    def update(self, context: Context, batch: Any, dataset_kind):
        """Update the metrics by passing the batch to ignite metric update method."""
        dataset = context.get_data_by_kind(dataset_kind)
        label = dataset.label_formatter(batch)
        prediction = context.infer(batch)
        for _, metric in self._state[dataset_kind]['scorers'].items():
            metric.update((prediction, label))

    def compute(self, context: Context) -> CheckResult:
        """Compute the metric result using the ignite metrics compute method and create display."""
        results = []
        for dataset_kind in [DatasetKind.TRAIN, DatasetKind.TEST]:
            dataset = context.get_data_by_kind(dataset_kind)
            metrics_df = metric_results_to_df(
                {k: m.compute() for k, m in self._state[dataset_kind]['scorers'].items()}, dataset
            )
            metrics_df['Dataset'] = dataset_kind.value
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
            fig.update_xaxes(tickprefix='Class Name', tickangle=60)

        fig = (
            fig.update_xaxes(title=None, type='category')
               .update_yaxes(title=None, matches=None)
               .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
               .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
        )

        return CheckResult(
            results_df,
            header='Class Performance',
            display=fig
        )

    def add_condition_test_performance_not_less_than(self: PR, min_score: float) -> PR:
        """Add condition - metric scores are not less than given score.

        Parameters
        ----------
        min_score : float
            Minimum score to pass the check.
        """
        def condition(check_result: pd.DataFrame):
            not_passed = check_result.loc[check_result['Value'] < min_score]
            not_passed_test = check_result.loc[check_result['Dataset'] == 'Test']
            if len(not_passed):
                details = f'Found metrics with scores below threshold:\n' \
                          f'{not_passed_test[["Class", "Metric", "Value"]].to_dict("records")}'
                return ConditionResult(False, details)
            return ConditionResult(True)

        return self.add_condition(f'Scores are not less than {min_score}', condition)

    def add_condition_train_test_relative_degradation_not_greater_than(self: PR, threshold: float = 0.1) -> PR:
        """Add condition that will check that test performance is not degraded by more than given percentage in train.

        Parameters
        ----------
        threshold : float
            maximum degradation ratio allowed (value between 0 and 1)
        """
        def _ratio_of_change_calc(score_1, score_2):
            if score_1 == 0:
                if score_2 == 0:
                    return 0
                return threshold + 1
            return (score_1 - score_2) / abs(score_1)

        def condition(check_result: pd.DataFrame) -> ConditionResult:
            test_scores = check_result.loc[check_result['Dataset'] == 'Test']
            train_scores = check_result.loc[check_result['Dataset'] == 'Train']

            if check_result.get('Class') is not None:
                classes = check_result['Class'].unique()
            else:
                classes = None
            explained_failures = []
            if classes is not None:
                for class_name in classes:
                    test_scores_class = test_scores.loc[test_scores['Class'] == class_name]
                    train_scores_class = train_scores.loc[train_scores['Class'] == class_name]
                    test_scores_dict = dict(zip(test_scores_class['Metric'], test_scores_class['Value']))
                    train_scores_dict = dict(zip(train_scores_class['Metric'], train_scores_class['Value']))
                    # Calculate percentage of change from train to test
                    diff = {score_name: _ratio_of_change_calc(score, test_scores_dict[score_name])
                            for score_name, score in train_scores_dict.items()}
                    failed_scores = [k for k, v in diff.items() if v > threshold]
                    if failed_scores:
                        for score_name in failed_scores:
                            explained_failures.append(f'{score_name} for class {class_name} '
                                                      f'(train={format_number(train_scores_dict[score_name])} '
                                                      f'test={format_number(test_scores_dict[score_name])})')
            else:
                test_scores_dict = dict(zip(test_scores['Metric'], test_scores['Value']))
                train_scores_dict = dict(zip(train_scores['Metric'], train_scores['Value']))
                # Calculate percentage of change from train to test
                diff = {score_name: _ratio_of_change_calc(score, test_scores_dict[score_name])
                        for score_name, score in train_scores_dict.items()}
                failed_scores = [k for k, v in diff.items() if v > threshold]
                if failed_scores:
                    for score_name in failed_scores:
                        explained_failures.append(f'{score_name}: '
                                                  f'train={format_number(train_scores_dict[score_name])}, '
                                                  f'test={format_number(test_scores_dict[score_name])}')
            if explained_failures:
                message = '\n'.join(explained_failures)
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Train-Test scores relative degradation is not greater than {threshold}',
                                  condition)

    def add_condition_class_performance_imbalance_ratio_not_greater_than(
        self: PR,
        threshold: float = 0.3,
        score: str = None
    ) -> PR:
        """Add condition.

        Verifying that relative ratio difference
        between highest-class and lowest-class is not greater than 'threshold'.

        Parameters
        ----------
        threshold : float
            ratio difference threshold
        score : str
            limit score for condition

        Returns
        -------
        Self
            instance of 'ClassPerformance' or it subtype

        Raises
        ------
        DeepchecksValueError
            if unknown score function name were passed;
        """
        # TODO: Redefine default scorers when making the condition work
        # if score is None:
        #     score = next(iter(MULTICLASS_SCORERS_NON_AVERAGE))

        def condition(check_result: pd.DataFrame) -> ConditionResult:
            if score not in set(check_result['Metric']):
                raise DeepchecksValueError(f'Data was not calculated using the scoring function: {score}')

            datasets_details = []
            for dataset in ['Test', 'Train']:
                data = check_result.loc[check_result['Dataset'] == dataset].loc[check_result['Metric'] == score]

                min_value_index = data['Value'].idxmin()
                min_row = data.loc[min_value_index]
                min_class_name = min_row['Class']
                min_value = min_row['Value']

                max_value_index = data['Value'].idxmax()
                max_row = data.loc[max_value_index]
                max_class_name = max_row['Class']
                max_value = max_row['Value']

                relative_difference = abs((min_value - max_value) / max_value)

                if relative_difference >= threshold:
                    details = (
                        f'Relative ratio difference between highest and lowest in {dataset} dataset '
                        f'classes is {format_percent(relative_difference)}, using {score} metric. '
                        f'Lowest class - {min_class_name}: {format_number(min_value)}; '
                        f'Highest class - {max_class_name}: {format_number(max_value)}'
                    )
                    datasets_details.append(details)
            if datasets_details:
                return ConditionResult(False, details='\n'.join(datasets_details))
            else:
                return ConditionResult(True)

        return self.add_condition(
            name=(
                f'Relative ratio difference between labels \'{score}\' score '
                f'is not greater than {format_percent(threshold)}'
            ),
            condition_func=condition
        )
