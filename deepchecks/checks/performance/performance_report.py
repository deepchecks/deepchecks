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
from typing import Callable, Dict
import pandas as pd
import plotly.graph_objects as go


from deepchecks.base.check import ModelComparisonContext
from deepchecks import CheckResult, Dataset, SingleDatasetBaseCheck, ConditionResult, ModelComparisonBaseCheck
from deepchecks.utils.metrics import get_scorers_dict, initialize_user_scorers, ModelType
from deepchecks.utils.validation import validate_model


__all__ = ['PerformanceReport', 'MultiModelPerformanceReport']


class PerformanceReport(SingleDatasetBaseCheck):
    """Summarize given scores on a dataset and model.

    Args:
        alternative_scorers (Dict[str, Callable], default None):
            An optional dictionary of scorer name to scorer functions.
            If none given, using default scorers
    """

    def __init__(self, alternative_scorers: Dict[str, Callable] = None):
        super().__init__()
        self.alternative_scorers = initialize_user_scorers(alternative_scorers)

    def run(self, dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            dataset (Dataset): a Dataset object
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult: value is dictionary in format 'score-name': score-value
        """
        return self._performance_report(dataset, model)

    def _performance_report(self, dataset: Dataset, model):
        Dataset.validate_dataset(dataset)
        dataset.validate_label()
        validate_model(dataset, model)

        # Get default scorers if no alternative, or validate alternatives
        scorers = get_scorers_dict(model, dataset, self.alternative_scorers)
        scores = {
            key: scorer(model, dataset)
            for key, scorer in scorers.items()
        }

        display_df = pd.DataFrame(scores.values(), columns=['Value'], index=scores.keys())
        display_df.index.name = 'Score'

        return CheckResult(scores, header='Performance Report', display=display_df)

    def add_condition_score_not_less_than(self, min_score: float):
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


class MultiModelPerformanceReport(ModelComparisonBaseCheck):
    """Summarize given scores between models on test datasets

    Args:
        alternative_scorers (Dict[str, Callable], default None):
            An optional dictionary of scorer name to scorer functions.
            If none given, using default scorers
    """

    def __init__(self, alternative_scorers: Dict[str, Callable] = None):
        super().__init__()
        self.alternative_scorers = initialize_user_scorers(alternative_scorers)

    def _run(self, context: ModelComparisonContext):
        first_model = context.models[0]
        first_test_ds = context.test_datasets[0]
        scorers = get_scorers_dict(first_model, first_test_ds, self.alternative_scorers, multiclass_avg=False)

        models_results = {}
        for _, test, model, model_name in context:
            curr_results = []
            for metric, scorer in scorers.items():
                score = scorer(model, test)
                # Multiclass scorers return numpy array of result per class
                if context.task_type == ModelType.MULTICLASS:
                    for class_i, value in enumerate(score):
                        curr_results.append([value, metric, class_i])
                else:
                    curr_results.append([score, metric])
            models_results[model_name] = curr_results

        # === Display ===
        if context.task_type == ModelType.MULTICLASS:
            fig = go.Figure()
            for model_name, results in models_results.items():
                display_df = pd.DataFrame(results, columns=['Value', 'Metric', 'Class'])
                display_df = display_df.sort_values(by=['Class', 'Metric'])
                # Edit classes names to add prefix 'Class '
                classes = display_df['Class'].apply(lambda x: f'Class {x}')
                fig.add_trace(go.Bar(
                    x=[classes, display_df['Metric']],
                    y=display_df['Value'],
                    name=model_name,
                ))
            fig.update_layout(bargap=0.4, bargroupgap=0)
        else:
            fig = go.Figure()
            for model_name, results in models_results.items():
                display_df = pd.DataFrame(results, columns=['Value', 'Metric'])
                values = display_df['Value'].apply(lambda x: abs(x))
                fig.add_trace(go.Bar(
                    x=display_df['Metric'],
                    y=values,
                    name=model_name,
                ))

        return CheckResult(models_results, display=[fig])
