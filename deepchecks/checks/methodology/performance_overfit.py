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
"""The train_validation_difference_overfit check module."""
import typing as t

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from deepchecks.utils.plot import colors
from deepchecks.utils.strings import format_percent
from deepchecks.utils.validation import validate_model
from deepchecks.utils.metrics import get_metrics_list
from deepchecks import (
    Dataset,
    CheckResult,
    TrainTestBaseCheck,
    ConditionResult
)


__all__ = ['TrainTestDifferenceOverfit']


TD = t.TypeVar('TD', bound='TrainTestDifferenceOverfit')


class TrainTestDifferenceOverfit(TrainTestBaseCheck):
    """Visualize overfit by displaying the difference between model metrics on train and on test data.

    The check would display the selected metrics for the training and test data, helping the user visualize
    the difference in performance between the two datasets. If no alternative_metrics are supplied, the check would
    use a list of default metrics. If they are supplied, alternative_metrics must be a dictionary, with the keys
    being metric names and the values being either a name of an sklearn scoring function
    (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring) or an sklearn scoring function.

    Args:
        alternative_metrics (Dict[str, Callable]): An optional dictionary of metric name or scorer functions
    """

    def __init__(
        self,
        alternative_metrics: t.Dict[str, t.Callable[[object, pd.DataFrame, str], float]] = None
    ):
        super().__init__()
        self.alternative_metrics = alternative_metrics

    def run(self, train_dataset: Dataset, test_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label column.
            test_dataset (Dataset): The test dataset object. Must contain a label column.
            model: A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult:
                value is a dataframe with metrics as indexes, and scores per training and test in the columns.
                data is a bar graph of the metrics for training and test data.

        Raises:
            DeepchecksValueError: If either of the dataset objects are not a Dataset instance with a label
        """
        return self._train_test_difference_overfit(train_dataset, test_dataset, model)

    def _train_test_difference_overfit(self, train_dataset: Dataset, test_dataset: Dataset, model,
                                       ) -> CheckResult:
        # Validate parameters
        Dataset.validate_dataset(train_dataset)
        Dataset.validate_dataset(test_dataset)
        train_dataset.validate_label()
        test_dataset.validate_label()
        train_dataset.validate_shared_label(test_dataset)
        train_dataset.validate_shared_features(test_dataset)
        validate_model(test_dataset, model)

        metrics = get_metrics_list(model, train_dataset, self.alternative_metrics)

        train_metrics = {key: scorer(model, train_dataset.data[train_dataset.features], train_dataset.label_col)
                         for key, scorer in metrics.items()}

        test_metrics = {key: scorer(model, test_dataset.data[test_dataset.features],
                                    test_dataset.label_col)
                        for key, scorer in metrics.items()}

        result = {'test': test_metrics, 'train': train_metrics}

        def plot_overfit():
            res_df = pd.DataFrame.from_dict({'Training Metrics': train_metrics, 'Test Metrics': test_metrics})
            width = 0.20
            indices = np.arange(len(res_df.index))

            plt.bar(indices, res_df['Training Metrics'].values.flatten(), width=width, color=colors['Train'])
            plt.bar(indices + width, res_df['Test Metrics'].values.flatten(), width=width, color=colors['Test'])
            plt.ylabel('Metrics')
            plt.xticks(ticks=indices + width / 2., labels=res_df.index)
            plt.xticks(rotation=30)
            plt.legend(res_df.columns, loc='upper right', bbox_to_anchor=(1.45, 1.02))

        return CheckResult(result, header='Train-Test Difference Overfit', display=[plot_overfit])

    def add_condition_difference_not_greater_than(self: TD, threshold: float) -> TD:
        """
        Add new condition.

        Add condition that will check that difference between train dataset metrics and test
        dataset metrics is not greater than X.

        Args:
            threshold: metrics difference upper bound
        """
        def condition(res: dict) -> ConditionResult:
            test_metrics = res['test']
            train_metrics = res['train']
            diff = {metric: score - test_metrics[metric] for metric, score in train_metrics.items()}
            failed_metrics = [k for k, v in diff.items() if v > threshold]
            if failed_metrics:
                explained_failures = []
                for metric in failed_metrics:
                    explained_failures.append(f'{metric} (train={format_percent(train_metrics[metric])} '
                                              f'test={format_percent(test_metrics[metric])})')
                message = f'Found performance degradation in: {", ".join(explained_failures)}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Train-Test metrics difference is not greater than {threshold}', condition)

    def add_condition_degradation_ratio_not_greater_than(self: TD, threshold: float = 0.1) -> TD:
        """
        Add new condition.

        Add condition that will check that train performance is not degraded by more than given percentage in test.

        Args:
            threshold: maximum degradation ratio allowed (value between 0 to 1)
        """
        def condition(res: dict) -> ConditionResult:
            test_metrics = res['test']
            train_metrics = res['train']
            # Calculate percentage of change from train to test
            diff = {metric: ((score - test_metrics[metric]) / score)
                    for metric, score in train_metrics.items()}
            failed_metrics = [k for k, v in diff.items() if v > threshold]
            if failed_metrics:
                explained_failures = []
                for metric in failed_metrics:
                    explained_failures.append(f'{metric} (train={format_percent(train_metrics[metric])} '
                                              f'test={format_percent(test_metrics[metric])})')
                message = f'Found performance degradation in: {", ".join(explained_failures)}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Train-Test metrics degradation ratio is not greater than {threshold}',
                                  condition)
