"""Utils module containing useful global functions."""
import enum
from typing import Any, Union
import sklearn
import catboost
from IPython import get_ipython
from sklearn.base import ClassifierMixin, RegressorMixin


__all__ = ['SUPPORTED_BASE_MODELS', 'MLChecksValueError', 'model_type_validation', 'is_notebook',
           'TaskType', 'task_type_check']

SUPPORTED_BASE_MODELS = [sklearn.base.BaseEstimator, catboost.CatBoost]


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
