"""The train_test_difference_overfit check module."""
from typing import Dict, Callable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from deepchecks.utils import model_type_validation
from deepchecks.metric_utils import get_metrics_list
from deepchecks import Dataset, CheckResult, TrainTestBaseCheck

__all__ = ['TrainTestDifferenceOverfit']


class TrainTestDifferenceOverfit(TrainTestBaseCheck):
    """Visualize overfit by displaying the difference between model metrics on train and on test data.

    The check would display the selected metrics for the training and test data, helping the user visualize
    the difference in performance between the two datasets. If no alternative_metrics are supplied, the check would
    use a list of default metrics. If they are supplied, alternative_metrics must be a dictionary, with the keys
    being metric names and the values being either a name of an sklearn scoring function
    (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring) or an sklearn scoring function.
    """

    def __init__(self, alternative_metrics: Dict[str, Callable] = None):
        """Initialize the TrainTestDifferenceOverfit check.

        Args:
            alternative_metrics (Dict[str, Callable]): An optional dictionary of metric name to scorer functions
        """
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
        func_name = self.__class__.__name__
        Dataset.validate_dataset(train_dataset, func_name)
        Dataset.validate_dataset(test_dataset, func_name)
        train_dataset.validate_label(func_name)
        test_dataset.validate_label(func_name)
        train_dataset.validate_shared_label(test_dataset, func_name)
        train_dataset.validate_shared_features(test_dataset, func_name)
        model_type_validation(model)

        metrics = get_metrics_list(model, train_dataset, self.alternative_metrics)

        train_metrics = {key: scorer(model, train_dataset.data[train_dataset.features()], train_dataset.label_col())
                         for key, scorer in metrics.items()}

        test_metrics = {key: scorer(model, test_dataset.data[test_dataset.features()],
                                    test_dataset.label_col())
                        for key, scorer in metrics.items()}

        res_df = pd.DataFrame.from_dict({'Training Metrics': train_metrics,
                                         'Test Metrics': test_metrics})

        def plot_overfit():
            width = 0.20
            my_cmap = plt.cm.get_cmap('Set2')
            indices = np.arange(len(res_df.index))

            colors = my_cmap(range(len(res_df.columns)))
            plt.bar(indices, res_df['Training Metrics'].values.flatten(), width=width, color=colors[0])
            plt.bar(indices + width, res_df['Test Metrics'].values.flatten(), width=width, color=colors[1])
            plt.ylabel('Metrics')
            plt.xticks(ticks=indices + width / 2., labels=res_df.index)
            plt.xticks(rotation=30)
            plt.legend(res_df.columns, loc='upper right', bbox_to_anchor=(1.45, 1.02))

        res = res_df.apply(lambda x: x[1] - x[0], axis=1)
        res.index = res.index.to_series().apply(lambda x: x + ' - Difference between Training data and Test data')

        return CheckResult(res, check=self.__class__, header='Train Test Difference Overfit',
                           display=[plot_overfit])
