"""Utils module containing useful global functions."""
import enum
from typing import Any, Union, Dict, Callable
import sklearn
import catboost
from IPython import get_ipython
from sklearn.base import ClassifierMixin, RegressorMixin


__all__ = ['SUPPORTED_BASE_MODELS', 'MLChecksValueError', 'model_type_validation', 'is_notebook',
           'TaskType', 'task_type_check', 'get_metrics_list']

from sklearn.metrics import get_scorer, make_scorer, accuracy_score, precision_score, recall_score, mean_squared_error

SUPPORTED_BASE_MODELS = [sklearn.base.BaseEstimator, catboost.CatBoost]

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


class TaskType(enum.Enum):
    regression = 'regression'  # regression
    binary = 'binary'  # binary classification
    multiclass = 'multiclass'  # multiclass classification


class MLChecksValueError(ValueError):
    """Exception class that represent a fault parameter was passed to MLChecks."""

    pass


def model_type_validation(model: Any):
    """Receive any object and check if it's an instance of a model we support.

    Raises
        MLChecksException: If the object is not of a supported type
    """
    if not any((isinstance(model, base) for base in SUPPORTED_BASE_MODELS)):
        raise MLChecksValueError(f'Model must inherit from one of supported models: {SUPPORTED_BASE_MODELS}')


def is_notebook():
    """Check if we're in an interactive context (Notebook, GUI support) or terminal-based.

    Returns:
        True if we are in a notebook context, False otherwise
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def model_dataset_shape_validation(model: Any, dataset: Any):
    """Check if number of dataset features matches the number model features.

    Raise:
        MLChecksException: if dataset does not match model
    """
    feature_count = None
    if isinstance(model, sklearn.base.BaseEstimator):
        feature_count = model.n_features_in_
    elif isinstance(model, catboost.CatBoost):
        feature_count = model.get_feature_count()

    if feature_count:
        if feature_count != len(dataset.features()):
            raise MLChecksValueError(f'model and dataset do not contain the same number of features:'
                                     f' model({feature_count}) dataset({len(dataset.features())})')
    else:
        raise MLChecksValueError('unable to extract number of features from model')


def task_type_check(model: Union[ClassifierMixin, RegressorMixin], dataset) -> TaskType:
    """Check task type (regression, binary, multiclass) according to model object and label column

    Args:
        model (Union[ClassifierMixin, RegressorMixin]): Model object - used to check if has predict_proba()
        dataset (Dataset): dataset - used to count the number of unique labels

    Returns:

    """

    model_type_validation(model)
    dataset.validate_label(task_type_check.__name__)

    if getattr(model, "predict_proba", None):
        model: ClassifierMixin
        if dataset.label_col().nunique() > 2:
            return TaskType.multiclass
        else:
            return TaskType.binary
    else:
        return TaskType.regression


def get_metrics_list(model, dataset: 'Dataset', alternative_metrics: Dict[str, Callable] = None):
    if alternative_metrics:
        metrics = {}
        for name, scorer in alternative_metrics.items():
            # If string, get scorer from sklearn. If callable, do heuristic to see if valid
            # Borrowed code from:
            # https://github.com/scikit-learn/scikit-learn/blob/844b4be24d20fc42cc13b957374c718956a0db39/sklearn/metrics/_scorer.py#L421
            if isinstance(scorer, str):
                metrics[name] = get_scorer(scorer)
            elif callable(scorer):
                # Heuristic to ensure user has not passed a metric
                module = getattr(scorer, "__module__", None)
                if (
                        hasattr(module, "startswith")
                        and module.startswith("sklearn.metrics.")
                        and not module.startswith("sklearn.metrics._scorer")
                        and not module.startswith("sklearn.metrics.tests.")
                ):
                    raise ValueError(
                        "scoring value %r looks like it is a metric "
                        "function rather than a scorer. A scorer should "
                        "require an estimator as its first parameter. "
                        "Please use `make_scorer` to convert a metric "
                        "to a scorer." % scorer
                    )
                metrics[name] = scorer
    else:
        # Check for model type
        model_type = task_type_check(model, dataset)
        if model_type == TaskType.binary:
            metrics = DEFAULT_BINARY_METRICS
        elif model_type == TaskType.multiclass:
            metrics = DEFAULT_MULTICLASS_METRICS
        elif model_type == TaskType.regression:
            metrics = DEFAULT_REGRESSION_METRICS
        else:
            raise(Exception('Inferred model_type is invalid'))

    return metrics
