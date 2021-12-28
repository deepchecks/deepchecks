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
from typing import Callable, Hashable, TypeVar, Dict, cast
import pandas as pd
import plotly.express as px

from deepchecks.base.check import ModelComparisonContext
from deepchecks.errors import DeepchecksValueError
from deepchecks.utils.strings import format_percent
from deepchecks import CheckResult, Dataset, TrainTestBaseCheck, ConditionResult, ModelComparisonBaseCheck
from deepchecks.utils.metrics import get_scorers_list, initialize_multi_scorers, ModelType, task_type_check
from deepchecks.utils.validation import validate_model


__all__ = ['PerformanceReport', 'MultiModelPerformanceReport']


PR = TypeVar('PR', bound='PerformanceReport')

class PerformanceReport(TrainTestBaseCheck):
    """Summarize given scores on a dataset and model.

    Args:
        alternative_scorers (Dict[str, Callable], default None):
            An optional dictionary of scorer name to scorer functions.
            If none given, using default scorers
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
        Dataset.validate_dataset(train_dataset)
        Dataset.validate_dataset(test_dataset)
        train_dataset.validate_label()
        test_dataset.validate_label()
        train_dataset.validate_shared_label(test_dataset)
        train_dataset.validate_shared_features(test_dataset)
        validate_model(test_dataset, model)

        task_type = task_type_check(model, train_dataset)

        # Get default scorers if no alternative, or validate alternatives
        scorers = get_scorers_list(model, test_dataset, self.alternative_scorers, multiclass_avg=False)
        datasets = {'Train': train_dataset, 'Test': test_dataset}
        if task_type == ModelType.MULTICLASS:
            x = ['Class', 'Dataset']
            results = []
            for dataset in datasets.keys():
                for scorer in scorers:
                    score_result = scorer(model, datasets[dataset])
                    # Multiclass scorers return numpy array of result per class
                    for class_i, value in enumerate(score_result):
                        if scorer.is_negative_scorer():
                            value = -value
                        results.append([dataset, value, scorer.name, class_i])
            results_df = pd.DataFrame(results, columns=['Dataset', 'Value', 'Metric', 'Class'])

        else:
            x = 'Dataset'
            results = []
            for dataset in datasets.keys():
                for scorer in scorers:
                    score_result = scorer(model, datasets[dataset])
                    if scorer.is_negative_scorer():
                        score_result = -score_result
                    results.append([dataset, score_result, scorer.name])

            results_df = pd.DataFrame(results, columns=['Dataset', 'Value', 'Metric'])
        fig = px.bar(results_df, x=x, y='Value', color='Dataset', barmode='group',
                        facet_col='Metric', facet_col_spacing=0.05)
        if task_type == ModelType.MULTICLASS:
            fig.update_xaxes(title=None, tickprefix='Class ', tickangle=60)
        else:
            fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
        fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))


        return CheckResult(results_df, header='Performance Report', display=fig)

    def add_condition_score_not_less_than(self: PR, min_score: float) -> PR:
        """Add condition - metric scores are not less than given score.

        Args:
            min_score (float): Minimal score to pass.
        """
        name = f'Score is not less than {min_score}'

        def condition(result, min_score):
            not_passed = {k: v for k, v in result.items() if v < min_score}
            if not_passed:
                details = f'Scores that did not pass threshold: {not_passed}'
                return ConditionResult(False, details)
            return ConditionResult(True)

        return self.add_condition(name, condition, min_score=min_score)
  
    def add_condition_difference_not_greater_than(self: PR, threshold: float) -> PR:
        """
        Add new condition.

        Add condition that will check that difference between train dataset scores and test
        dataset scores is not greater than X.

        Args:
            threshold: scores difference upper bound
        """
        def condition(res: dict) -> ConditionResult:
            test_scores = res['test']
            train_scores = res['train']
            diff = {score_name: score - test_scores[score_name] for score_name, score in train_scores.items()}
            failed_scores = [k for k, v in diff.items() if v > threshold]
            if failed_scores:
                explained_failures = []
                for score_name in failed_scores:
                    explained_failures.append(f'{score_name} (train={format_percent(train_scores[score_name])} '
                                              f'test={format_percent(test_scores[score_name])})')
                message = f'Found performance degradation in: {", ".join(explained_failures)}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Train-Test scores difference is not greater than {threshold}', condition)

    def add_condition_degradation_ratio_not_greater_than(self: PR, threshold: float = 0.1) -> PR:
        """
        Add new condition.

        Add condition that will check that train performance is not degraded by more than given percentage in test.

        Args:
            threshold: maximum degradation ratio allowed (value between 0 to 1)
        """
        def condition(res: dict) -> ConditionResult:
            test_scores = res['test']
            train_scores = res['train']
            # Calculate percentage of change from train to test
            diff = {score_name: ((score - test_scores[score_name]) / score)
                    for score_name, score in train_scores.items()}
            failed_scores = [k for k, v in diff.items() if v > threshold]
            if failed_scores:
                explained_failures = []
                for score_name in failed_scores:
                    explained_failures.append(f'{score_name} (train={format_percent(train_scores[score_name])} '
                                              f'test={format_percent(test_scores[score_name])})')
                message = f'Found performance degradation in: {", ".join(explained_failures)}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Train-Test scores degradation ratio is not greater than {threshold}',
                                  condition)


    def add_condition_ratio_difference_not_greater_than(
        self: PR,
        threshold: float = 0.3,
        score: str = 'F1'
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
        scorers = self.alternative_scorers or self._default_scorers
        scorers = set(scorers.keys())

        if score not in scorers:
            raise DeepchecksValueError(f'Data was not calculated using the scoring function: {score}')

        def condition(check_result: Dict[str, Dict[Hashable, float]]) -> ConditionResult:
            datasets_details = []
            for dataset in ['Test', 'Train']:
                data = cast(
                    Dict[str, Dict[Hashable, float]],
                    pd.DataFrame.from_dict(check_result).transpose().to_dict()
                )
                data = [
                    classes_values
                    for score_name, classes_values in data.items()
                    if score_name == score
                ]

                if len(data) == 0:
                    raise DeepchecksValueError(f'Expected that check result will contain next score - {score}')

                classes_values = next(iter(data))
                getval = lambda it: it[1]
                lowest_class_name, min_value = min(classes_values.items(), key=getval)
                highest_class_name, max_value = max(classes_values.items(), key=getval)
                relative_difference = abs((min_value - max_value) / max_value)

                if relative_difference >= threshold:
                    details = (
                        f'Relative ratio difference between highest and lowest in {dataset} dataset'
                        f'classes is greater than {format_percent(threshold)}. '
                        f'Score: {score}, lowest class: {lowest_class_name}, highest class: {highest_class_name};'
                    )
                    datasets_details.append(details)
            if datasets_details:
                return ConditionResult(False, details='\n'.join(datasets_details))
            else:
                return ConditionResult(True)

        return self.add_condition(
            name=(
                f"Relative ratio difference between labels '{score}' score "
                f"is not greater than {format_percent(threshold)}"
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

        if context.task_type == ModelType.MULTICLASS:
            x = ['Class', 'Model']
            results = []
            for _, test, model, model_name in context:
                for scorer in scorers:
                    score_result = scorer(model, test)
                    # Multiclass scorers return numpy array of result per class
                    for class_i, value in enumerate(score_result):
                        if scorer.is_negative_scorer():
                            value = -value
                        results.append([model_name, value, scorer.name, class_i])
            results_df = pd.DataFrame(results, columns=['Model', 'Value', 'Metric', 'Class'])
        else:
            x = 'Model'
            results = []
            for _, test, model, model_name in context:
                for scorer in scorers:
                    score_result = scorer(model, test)
                    if scorer.is_negative_scorer():
                        score_result = -score_result
                    results.append([model_name, score_result, scorer.name])

            results_df = pd.DataFrame(results, columns=['Model', 'Value', 'Metric'])

        fig = px.bar(results_df, x=x, y='Value', color='Model', barmode='group',
                        facet_col='Metric', facet_col_spacing=0.05)
        if context.task_type == ModelType.MULTICLASS:
            fig.update_xaxes(title=None, tickprefix='Class ', tickangle=60)
        else:
            fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
        fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))

        return CheckResult(results_df, display=[fig])
