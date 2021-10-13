from typing import List, Dict, Callable

import matplotlib.pyplot as plt
import pandas as pd

from mlchecks.display import format_check_display
from mlchecks.utils import model_type_validation, MLChecksValueError, task_type_check, TaskType, get_plt_html_str
from mlchecks import validate_dataset, Dataset, CheckResult
from sklearn.metrics import make_scorer, precision_score, recall_score, mean_squared_error, accuracy_score

__all__ = ['train_validation_difference_overfit']

DEFAULT_BINARY_METRICS = {
    'Accuracy': make_scorer(accuracy_score),
    'Precision': make_scorer(precision_score),
    'Recall': make_scorer(recall_score)
}

DEFAULT_MULTICLASS_METRICS = {
    'Accuracy': make_scorer(accuracy_score),
    'Precision - Macro Average': make_scorer(precision_score, average='macro'),
    'Recall - Macro Average': make_scorer(recall_score, average='macro')
}

DEFAULT_REGRESSION_METRICS = {
    'RMSE': make_scorer(mean_squared_error, squared=False),
    'MSE': make_scorer(mean_squared_error),
}


def calculate_metrics(train_dataset: Dataset, validation_dataset: Dataset, model,
                      metrics: Dict[str, Callable[[pd.Series, pd.Series], float]]) -> CheckResult:
    """Calculates overfit result for a dict of metric scoring functions

    Args:
        train_dataset ():
        validation_dataset ():
        model ():
        metrics ():

    Returns:

    """
    train_metrics = {key: scorer(model, train_dataset[train_dataset.features()], train_dataset.label_col())
                     for key, scorer in metrics.items()}

    val_metrics = {key: scorer(model, validation_dataset[validation_dataset.features()], validation_dataset.label_col())
                   for key, scorer in metrics.items()}

    diff_in_metrics = {key: val_metrics[key] - train_metrics[key] for key in metrics}

    res_df = pd.DataFrame.from_dict({'Difference in metric between validation and Train': diff_in_metrics})

    res_df.plot.bar(figsize=(15, 10))
    plt.xticks(rotation=30)

    return CheckResult(diff_in_metrics,
                       display={'text/html': format_check_display('Train Validation Difference Overfit',
                                                                  train_validation_difference_overfit, get_plt_html_str())})


def train_validation_difference_overfit(train_dataset: Dataset, validation_dataset: Dataset, model,
                                        additional_metrics: List[str] = None) -> CheckResult:
    """Check for overfit by checking the difference between model metrics on train and on validation data.

    Args:
        train_dataset
        validation_dataset
        model
        additional_metrics
    """
    # Validate parameters
    func_name = 'performance_overfit'
    model_type_validation(model)
    validate_dataset(train_dataset, func_name)
    validate_dataset(validation_dataset, func_name)
    train_dataset.validate_shared_label(validation_dataset, func_name)
    train_dataset.validate_shared_features(validation_dataset, func_name)

    # TODO: actually use additional_metrics

    # Check for model type
    model_type = task_type_check(model, train_dataset)
    if model_type == TaskType.binary:
        metrics = DEFAULT_BINARY_METRICS
    elif model_type == TaskType.multiclass:
        metrics = DEFAULT_MULTICLASS_METRICS
    elif model_type == TaskType.regression:
        metrics = DEFAULT_REGRESSION_METRICS
    else:
        raise(Exception('Inferred model_type is invalid'))

    return calculate_metrics(train_dataset, validation_dataset, model, metrics)
