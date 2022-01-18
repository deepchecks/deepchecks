# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing performance report check."""
from typing import Callable, TypeVar, Dict, cast
import pandas as pd
import plotly.express as px

from deepchecks import CheckResult, Dataset, TrainTestBaseCheck, ConditionResult, ModelComparisonBaseCheck
from deepchecks.base.check import ModelComparisonContext
from deepchecks.errors import DeepchecksValueError
from deepchecks.utils.strings import format_percent, format_number
from deepchecks.utils.validation import validate_model
from deepchecks.utils.metrics import (
    MULTICLASS_SCORERS_NON_AVERAGE,
    get_scorers_list,
    initialize_multi_scorers,
    ModelType,
    task_type_check
)


__all__ = ['PerformanceReport', 'MultiModelPerformanceReport']


PR = TypeVar('PR', bound='PerformanceReport')


class PerformanceReport(TrainTestBaseCheck):
    """Summarize given scores on a dataset and model.

    Args:
        alternative_scorers (Dict[str, Callable], default None):
            An optional dictionary of scorer name to scorer functions.
            If none given, using default scorers

    Notes
    -----
    Scorers are a convention of sklearn to evaluate a model.
    `See scorers documentation <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`_
    A scorer is a function which accepts (model, X, y_true) and returns a float result which is the score.
    For every scorer higher scores are better than lower scores.

    You can create a scorer out of existing sklearn metrics:

    .. code-block:: python

        from sklearn.metrics import roc_auc_score, make_scorer
        auc_scorer = make_scorer(roc_auc_score)

    Or you can implement your own:

    .. code-block:: python

        from sklearn.metrics import make_scorer


        def my_mse(y_true, y_pred):
            return (y_true - y_pred) ** 2


        # Mark greater_is_better=False, since scorers always suppose to return
        # value to maximize.
        my_mse_scorer = make_scorer(my_mse, greater_is_better=False)
    """

    def __init__(self, alternative_scorers: Dict[str, Callable] = None):
        super().__init__()
        self.alternative_scorers = initialize_multi_scorers(alternative_scorers)

    def run(self, train_dataset: Dataset, test_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            dataset (Dataset): a Dataset object
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult: value is dictionary in format 'score-name': score-value
        """
        return self._performance_report(train_dataset, test_dataset, model)

    def _performance_report(self, train_dataset: Dataset, test_dataset: Dataset, model):
        train_dataset = Dataset.ensure_not_empty_dataset(train_dataset)
        test_dataset = Dataset.ensure_not_empty_dataset(test_dataset)

        self._datasets_share_label([train_dataset, test_dataset])
        self._datasets_share_features([train_dataset, test_dataset])
        validate_model(test_dataset, model)

        task_type = task_type_check(model, train_dataset)
        clasess = train_dataset.classes

        # Get default scorers if no alternative, or validate alternatives
        scorers = get_scorers_list(model, test_dataset, self.alternative_scorers, multiclass_avg=False)
        datasets = {'Train': train_dataset, 'Test': test_dataset}

        if task_type in {ModelType.MULTICLASS, ModelType.BINARY}:
            plot_x_axis = 'Class'
            results = []

            for dataset_name, dataset in datasets.items():
                label = cast(pd.Series, dataset.label_col)
                n_samples = label.groupby(label).count()
                results.extend(
                    [dataset_name, class_name, scorer.name, class_score, n_samples[class_name]]
                    for scorer in scorers
                    # scorer returns numpy array of results with item per class
                    for class_score, class_name in zip(scorer(model, dataset), clasess)
                )

            results_df = pd.DataFrame(results, columns=['Dataset', 'Class', 'Metric', 'Value', 'Number of samples'])

        else:
            plot_x_axis = 'Dataset'
            results = [
                [dataset_name, scorer.name, scorer(model, dataset), cast(pd.Series, dataset.label_col).count()]
                for dataset_name, dataset in datasets.items()
                for scorer in scorers
            ]
            results_df = pd.DataFrame(results, columns=['Dataset', 'Metric', 'Value', 'Number of samples'])

        fig = px.histogram(
            results_df,
            x=plot_x_axis,
            y='Value',
            color='Dataset',
            barmode='group',
            facet_col='Metric',
            facet_col_spacing=0.05,
            hover_data=['Number of samples']
        )

        if task_type in [ModelType.MULTICLASS, ModelType.BINARY]:
            fig.update_xaxes(tickprefix='Class ', tickangle=60)

        fig = (
            fig.update_xaxes(title=None, type='category')
            .update_yaxes(title=None, matches=None)
            .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
            .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
        )

        return CheckResult(
            results_df,
            header='Performance Report',
            display=fig
        )

    def add_condition_test_performance_not_less_than(self: PR, min_score: float) -> PR:
        """Add condition - metric scores are not less than given score.

        Args:
            min_score (float): Minimal score to pass.
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

        Args:
            threshold: maximum degradation ratio allowed (value between 0 and 1)
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

        Args:
            threshold: ratio difference threshold
            score: limit score for condition

        Returns:
            Self: instance of 'ClassPerformance' or it subtype

        Raises:
            DeepchecksValueError:
                if unknown score function name were passed;
        """
        if score is None:
            score = next(iter(MULTICLASS_SCORERS_NON_AVERAGE))

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


class MultiModelPerformanceReport(ModelComparisonBaseCheck):
    """Summarize performance scores for multiple models on test datasets.

    Args:
        alternative_scorers (Dict[str, Callable], default None):
            An optional dictionary of scorer name to scorer functions.
            If none given, using default scorers
    """

    def __init__(self, alternative_scorers: Dict[str, Callable] = None):
        super().__init__()
        self.alternative_scorers = initialize_multi_scorers(alternative_scorers)

    def run_logic(self, context: ModelComparisonContext):
        """Run check logic."""
        first_model = context.models[0]
        first_test_ds = context.test_datasets[0]
        scorers = get_scorers_list(first_model, first_test_ds, self.alternative_scorers, multiclass_avg=False)

        if context.task_type in [ModelType.MULTICLASS, ModelType.BINARY]:
            plot_x_axis = ['Class', 'Model']
            results = []

            for _, test_dataset, model, model_name in context:
                label = cast(pd.Series, test_dataset.label_col)
                n_samples = label.groupby(label).count()
                results.extend(
                    [model_name, class_score, scorer.name, class_name, n_samples[class_name]]
                    for scorer in scorers
                    # scorer returns numpy array of results with item per class
                    for class_score, class_name in zip(scorer(model, test_dataset), test_dataset.classes)
                )

            results_df = pd.DataFrame(results, columns=['Model', 'Value', 'Metric', 'Class', 'Number of samples'])

        else:
            plot_x_axis = 'Model'
            results = [
                [model_name, scorer(model, test_dataset), scorer.name, cast(pd.Series, test_dataset.label_col).count()]
                for _, test_dataset, model, model_name in context
                for scorer in scorers
            ]
            results_df = pd.DataFrame(results, columns=['Model', 'Value', 'Metric', 'Number of samples'])

        fig = px.histogram(
            results_df,
            x=plot_x_axis,
            y='Value',
            color='Model',
            barmode='group',
            facet_col='Metric',
            facet_col_spacing=0.05,
            hover_data=['Number of samples'],
        )

        if context.task_type in [ModelType.MULTICLASS, ModelType.BINARY]:
            fig.update_xaxes(title=None, tickprefix='Class ', tickangle=60)
        else:
            fig.update_xaxes(title=None)

        fig = (
            fig.update_yaxes(title=None, matches=None)
            .for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
            .for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
        )

        return CheckResult(results_df, display=[fig])
