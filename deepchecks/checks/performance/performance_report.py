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
from deepchecks import CheckResult, Dataset, SingleDatasetBaseCheck, ConditionResult
from deepchecks.utils.metrics import get_metrics_list
from deepchecks.utils.validation import validate_model


__all__ = ['PerformanceReport']


class PerformanceReport(SingleDatasetBaseCheck):
    """Summarize given scores on a dataset and model.

    Args:
        alternative_scorers (Dict[str, Callable], default None):
            An optional dictionary of scorer name to scorer functions.
            If none given, using default scorers
    """

    def __init__(self, alternative_scorers: Dict[str, Callable] = None):
        super().__init__()
        self.alternative_scorers = alternative_scorers

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
        scorers = get_metrics_list(model, dataset, self.alternative_scorers)
        scores = {
            key: scorer(model, dataset.features_columns, dataset.label_col)
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
                details = f'Scores that are less than {min_score}: {not_passed}'
                return ConditionResult(False, details)
            return ConditionResult(True)

        return self.add_condition(name, condition, min_score=min_score)
