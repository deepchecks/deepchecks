"""The train_validation_difference_overfit check module."""
from typing import Dict, Callable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from mlchecks.utils import model_type_validation, get_metrics_list
from mlchecks import Dataset, CheckResult, TrainValidationBaseCheck

__all__ = ['train_validation_difference_overfit', 'TrainValidationDifferenceOverfit']


def calculate_metrics(train_dataset: Dataset, validation_dataset: Dataset, model,
                      metrics: Dict[str, Callable]) -> CheckResult:
    """Calculate overfit result for a dict of metric scoring functions.

    Args:
        train_dataset (Dataset): The training dataset object. Must contain a label column.
        validation_dataset (Dataset): The validation dataset object. Must contain a label column.
        model: A scikit-learn-compatible fitted estimator instance
        metrics (Dict[str, Callable]): An dictionary of metric name to scorer functions

    Returns:
        Dict of
    """
    train_metrics = {key: scorer(model, train_dataset.data[train_dataset.features()], train_dataset.label_col())
                     for key, scorer in metrics.items()}

    val_metrics = {key: scorer(model, validation_dataset.data[validation_dataset.features()],
                               validation_dataset.label_col())
                   for key, scorer in metrics.items()}

    res_df = pd.DataFrame.from_dict({'Training Metrics': train_metrics,
                                     'Validation Metrics': val_metrics})

    def plot_overfit():
        width = 0.20
        my_cmap = plt.cm.get_cmap('Set2')
        indices = np.arange(len(res_df.index))

        colors = my_cmap(range(len(res_df.columns)))
        plt.bar(indices, res_df['Training Metrics'].values.flatten(), width=width, color=colors[0])
        plt.bar(indices + width, res_df['Validation Metrics'].values.flatten(), width=width, color=colors[1])
        plt.ylabel('Metrics')
        plt.xticks(ticks=indices + width / 2., labels=res_df.index)
        plt.xticks(rotation=30)
        plt.legend(res_df.columns, loc='upper right', bbox_to_anchor=(1.45, 1.02))

    res = res_df.apply(lambda x: x[1] - x[0], axis=1)
    res.index = res.index.to_series().apply(lambda x: x + ' - Difference between Training data and Validation data')

    return CheckResult(res, check=calculate_metrics,
                       display=[plot_overfit])


def train_validation_difference_overfit(train_dataset: Dataset, validation_dataset: Dataset, model,
                                        alternative_metrics: Dict[str, Callable] = None) -> CheckResult:
    """Visualize overfit by displaying the difference between model metrics on train and on validation data.

    The check would display the selected metrics for the training and validation data, helping the user visualize
    the difference in performance between the two datasets. If no alternative_metrics are supplied, the check would
    use a list of default metrics. If they are supplied alternative_metrics must be a dictionary, with the keys
    being metrics names and the values being either string of sklearn scoring functions
    (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring) or sklearn scoring functions.

    Args:
        train_dataset (Dataset): The training dataset object. Must contain a label column.
        validation_dataset (Dataset): The validation dataset object. Must contain a label column.
        model: A scikit-learn-compatible fitted estimator instance
        alternative_metrics (Dict[str, Callable]): An optional dictionary of metric name to scorer functions

    Returns:
        CheckResult:
            value is a dataframe with metrics as indexes, and scores per training and validation in the columns.
            data is a bar graph of the metrics for training and validation data.

    Raises:
        MLChecksValueError: If the object is not a Dataset instance with a label
    """
    # Validate parameters
    func_name = train_validation_difference_overfit.__name__
    Dataset.validate_dataset(train_dataset, func_name)
    Dataset.validate_dataset(validation_dataset, func_name)
    train_dataset.validate_label(func_name)
    validation_dataset.validate_label(func_name)
    train_dataset.validate_shared_label(validation_dataset, func_name)
    train_dataset.validate_shared_features(validation_dataset, func_name)
    model_type_validation(model)

    metrics = get_metrics_list(model, train_dataset, alternative_metrics)

    return calculate_metrics(train_dataset, validation_dataset, model, metrics)


class TrainValidationDifferenceOverfit(TrainValidationBaseCheck):
    """Check if validation dates are present in train data."""

    def run(self, train_dataset: Dataset, validation_dataset: Dataset, model=None) -> CheckResult:
        """Visualize overfit by displaying the difference between model metrics on train and on validation data.

        The check would display the selected metrics for the training and validation data, helping the user visualize
        the difference in performance between the two datasets. If no alternative_metrics are supplied, the check would
        use a list of default metrics. If they are supplied alternative_metrics must be a dictionary, with the keys
        being metrics names and the values being either string of sklearn scoring functions
        (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring) or sklearn scoring functions.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label column.
            validation_dataset (Dataset): The validation dataset object. Must contain a label column.
            model: A scikit-learn-compatible fitted estimator instance
            alternative_metrics (Dict[str, Callable]): An optional dictionary of metric name to scorer functions

        Returns:
            CheckResult:
                value is a dataframe with metrics as indexes, and scores per training and validation in the columns.
                data is a bar graph of the metrics for training and validation data.

        Raises:
            MLChecksValueError: If the object is not a Dataset instance with a label
        """
        return train_validation_difference_overfit(train_dataset, validation_dataset, model,
                                                   alternative_metrics=self.params.get('alternative_metrics'))
