"""Utils module containing useful global functions."""
from typing import Any
import sklearn
from IPython import get_ipython


__all__ = ['DeepchecksValueError', 'model_type_validation', 'is_notebook']


class DeepchecksValueError(ValueError):
    """Exception class that represent a fault parameter was passed to Deepchecks."""

    pass


def model_type_validation(model: Any):
    """Receive any object and check if it's an instance of a model we support.

    Raises
        DeepchecksValueError: If the object is not of a supported type
    """
    supported_by_class_name = ['CatBoostClassifier', 'CatBoostRegressor']
    supported_by_class_instance = (sklearn.base.BaseEstimator,)
    if isinstance(model, supported_by_class_instance) or model.__class__.__name__ in supported_by_class_name:
        return
    else:
        raise DeepchecksValueError('Model must inherit from one of supported models: sklearn.base.BaseEstimator or '
                                 'CatBoost')


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
