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
    """Summarize given metrics on a dataset and model.

    Args:
        alternative_metrics (Dict[str, Callable]): An optional dictionary of metric name or scorer functions.
        If none given, using default metrics
    """

    def __init__(self, alternative_metrics: Dict[str, Callable] = None):
        super().__init__()
        self.alternative_metrics = alternative_metrics

    def run(self, dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            dataset (Dataset): a Dataset object
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult: value is dictionary in format 'metric': score
        """
        return self._performance_report(dataset, model)

    def _performance_report(self, dataset: Dataset, model):
        Dataset.validate_dataset(dataset)
        dataset.validate_label()
        validate_model(dataset, model)

        # Get default metrics if no alternative, or validate alternatives
        metrics = get_metrics_list(model, dataset, self.alternative_metrics)
        scores = {key: scorer(model, dataset.features_columns, dataset.label_col) for key, scorer in
                  metrics.items()}

        display_df = pd.DataFrame(scores.values(), columns=['Score'], index=scores.keys())
        display_df.index.name = 'Metric'

        return CheckResult(scores, header='Performance Report', display=display_df)

    def add_condition_score_not_less_than(self, min_score: float):
        """Add condition - metric scores are not less than given score.

        Args:
            min_score (float): Minimal score to pass.
        """
        name = f'Metrics score is not less than {min_score}'

        def condition(result, min_score):
            not_passed = {k: v for k, v in result.items() if v < min_score}
            if not_passed:
                details = f'Metrics with lower score: {not_passed}'
                return ConditionResult(False, details)
            return ConditionResult(True)

        return self.add_condition(name, condition, min_score=min_score)
