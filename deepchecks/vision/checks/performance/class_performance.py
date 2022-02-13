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
from typing import TypeVar, List, Union, Any

import pandas as pd
import plotly.express as px
from ignite.metrics import Metric

from deepchecks.core import CheckResult, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.strings import format_percent, format_number
from deepchecks.vision import TrainTestCheck, Context
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.metrics_utils.metrics import get_scorers_list
from deepchecks.vision.utils import ClassificationPredictionFormatter, DetectionPredictionFormatter

__all__ = ['ClassPerformance']

PR = TypeVar('PR', bound='ClassPerformance')


class ClassPerformance(TrainTestCheck):
    """Summarize given metrics on a dataset and model.

    Parameters
    ----------
    alternative_metrics : List[Metric], default: None
        A list of ignite.Metric objects whose score should be used. If None are given, use the default metrics.
    prediction_formatter : Union[ClassificationPredictionFormatter, DetectionPredictionFormatter, None], default: None
        An encoder to convert predictions to a format that can be used by the metrics.
    """

    def __init__(self,
                 alternative_metrics: List[Metric] = None,
                 prediction_formatter: Union[ClassificationPredictionFormatter, DetectionPredictionFormatter] = None):
        super().__init__()
        self.alternative_metrics = alternative_metrics
        self.prediction_formatter = prediction_formatter
        self._state = {}

    def initialize_run(self, context: Context):
        """Initialize run by creating the _state member with metrics for train and test."""
        context.assert_task_type(TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION)

        self._state = {'train': {}, 'test': {}}
        self._state['train']['scorers'] = get_scorers_list(context.train, self.alternative_metrics)
        self._state['test']['scorers'] = get_scorers_list(context.train, self.alternative_metrics)
        for dataset_name in ['train', 'test']:
            for _, metric in self._state[dataset_name]['scorers'].items():
                metric.reset()

    def update(self, context: Context, batch: Any, dataset_name: str = 'train'):
        """Update the metrics by passing the batch to ignite metric update method."""
        if dataset_name == 'train':
            dataset = context.train
        else:
            dataset = context.test
        images = batch[0]
        label = dataset.label_transformer(batch[1])
        prediction = self.prediction_formatter(context.infer(images))
        for _, metric in self._state[dataset_name]['scorers'].items():
            metric.update((prediction, label))

    def compute(self, context: Context) -> CheckResult:
        """Compute the metric result using the ignite metrics compute method and create display."""
        self._state['train']['n_samples'] = context.train.get_samples_per_class()
        self._state['test']['n_samples'] = context.test.get_samples_per_class()
        self._state['classes'] = sorted(context.train.get_samples_per_class().keys())

        results = []
        for dataset_name in ['train', 'test']:
            n_samples = self._state[dataset_name]['n_samples']
            results.extend(
                [dataset_name, class_name, name, class_score, n_samples[class_name]]
                for name, score in [(name, metric.compute().tolist()) for name, metric in
                                    self._state[dataset_name]['scorers'].items()]
                # scorer returns numpy array of results with item per class
                for class_score, class_name in zip(score, self._state['classes'])
            )

        results_df = pd.DataFrame(results, columns=['Dataset', 'Class', 'Metric', 'Value', 'Number of samples']
                                  ).sort_values(by=['Class'])

        fig = px.histogram(
            results_df,
            x='Class',
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
