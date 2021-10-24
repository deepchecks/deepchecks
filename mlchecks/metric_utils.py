"""Utils module containing utilities for checks working with metrics."""
import enum
from numbers import Number
from typing import Union, Dict, Callable
from sklearn.metrics import get_scorer, make_scorer, accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.base import ClassifierMixin, RegressorMixin

__all__ = ['ModelType', 'task_type_check', 'get_metrics_list']

from mlchecks.utils import model_type_validation

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


class ModelType(enum.Enum):
    """Enum containing suppoerted task types."""

    REGRESSION = 'regression'  # regression
    BINARY = 'binary'  # binary classification
    MULTICLASS = 'multiclass'  # multiclass classification


DEFAULT_METRICS_DICT = {
    ModelType.BINARY: DEFAULT_BINARY_METRICS,
    ModelType.MULTICLASS: DEFAULT_MULTICLASS_METRICS,
    ModelType.REGRESSION: DEFAULT_REGRESSION_METRICS
}


def task_type_check(model: Union[ClassifierMixin, RegressorMixin], dataset: 'Dataset') -> ModelType:
    """Check task type (regression, binary, multiclass) according to model object and label column.

    Args:
        model (Union[ClassifierMixin, RegressorMixin]): Model object - used to check if has predict_proba()
        dataset (Dataset): dataset - used to count the number of unique labels

    Returns:
        TaskType enum corresponding to the model and dataset
    """
    model_type_validation(model)
    dataset.validate_label(task_type_check.__name__)

    if getattr(model, 'predict_proba', None):
        model: ClassifierMixin
        if dataset.label_col().nunique() > 2:
            return ModelType.MULTICLASS
        else:
            return ModelType.BINARY
    else:
        return ModelType.REGRESSION


def get_metrics_list(model, dataset: 'Dataset', alternative_metrics: Dict[str, Callable] = None
                     ) -> Dict[str, Callable]:
    """Return list of scorer objects to use in a metrics-dependant check.

    If no alternative_metrics is supplied, then a default list of metrics is used per task type, as it is inferred
    from the dataset and model. If a list is supplied, then the scorer functions are checked and used instead.

    Args:
        model (BaseEstimator): Model object for which the metrics would be calculated
        dataset (Dataset): Dataset object on which the metrics would be calculated
        alternative_metrics (Dict[str, Callable]): Optional dictionary of sklearn scorers to use instead of default list

    Returns:
        Dictionary containing names of metrics and scorer functions for the metrics.
    """
    if alternative_metrics:
        metrics = {}
        for name, scorer in alternative_metrics.items():
            # If string, get scorer from sklearn. If callable, do heuristic to see if valid
            # Borrowed code from:
            # https://github.com/scikit-learn/scikit-learn/blob/844b4be24d20fc42cc13b957374c718956a0db39/sklearn/metrics/_scorer.py#L421
            if isinstance(scorer, str):
                metrics[name] = get_scorer(scorer)
            elif callable(scorer):
                # Check that scorer runs for given model and data
                assert isinstance(scorer(model, dataset.data[dataset.features()].head(2), dataset.label_col().head(2)),
                                  Number)
                metrics[name] = scorer
    else:
        # Check for model type
        model_type = task_type_check(model, dataset)
        metrics = DEFAULT_METRICS_DICT[model_type]

    return metrics
