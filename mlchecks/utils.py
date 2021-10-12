"""Utils module containing useful global functions."""
import base64
import io
from typing import Any
import sklearn
import matplotlib.pyplot as plt
import catboost
from IPython import get_ipython

__all__ = ['SUPPORTED_BASE_MODELS', 'MLChecksValueError', 'model_type_validation', 'is_notebook']


SUPPORTED_BASE_MODELS = [sklearn.base.BaseEstimator, catboost.CatBoost]


class MLChecksValueError(ValueError):
    pass


def get_plt_base64():
    """
    Returns:
        string of base64 encoding of the matplotlib.pyplot graph
    """
    plt_buffer = io.BytesIO()
    plt.savefig(plt_buffer, format='jpg')
    plt_buffer.seek(0)
    return base64.b64encode(plt_buffer.read()).decode("utf-8")


def get_plt_html_str():
    """
    Returns:
        string in text/html format in order to display the plot in html
    """
    jpg = get_plt_base64()
    return f'<img src="data:image/jpg;base64, {jpg}"/>'


def model_type_validation(model: Any):
    """Receive any object and check if it's an instance of a model we support

    Raises
        MLChecksException: If the object is not of a supported type
    """
    if not any([isinstance(model, base) for base in SUPPORTED_BASE_MODELS]):
        raise MLChecksValueError(f'Model must inherit from one of supported models: {SUPPORTED_BASE_MODELS}')


def is_notebook():
    """
    check if we're in an interactive context (Notebook, GUI support) or terminal-based.

    Returns:
        True if we are in a notebook context, False otherwise
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter