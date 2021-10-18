"""Utils module containing useful global functions."""
import base64
import io
from typing import Any, List, Iterable, Union
import sklearn
import matplotlib.pyplot as plt
import catboost
from IPython import get_ipython

__all__ = ['SUPPORTED_BASE_MODELS', 'MLChecksValueError', 'model_type_validation', 'is_notebook', 'get_plt_html_str',
           'get_txt_html_str']

SUPPORTED_BASE_MODELS = [sklearn.base.BaseEstimator, catboost.CatBoost]


class MLChecksValueError(ValueError):
    """Exception class that represent a fault parameter was passed to MLChecks."""

    pass


def get_plt_base64():
    """Convert plot to base64.

    Returns:
        string of base64 encoding of the matplotlib.pyplot graph
    """
    plt_buffer = io.BytesIO()
    plt.style.use('seaborn')
    plt.savefig(plt_buffer, format='jpg', bbox_inches='tight')
    plt_buffer.seek(0)
    return base64.b64encode(plt_buffer.read()).decode('utf-8')


def get_plt_html_str() -> str:
    """Convert plot to html image tag.

    Returns:
        string in text/html format in order to display the plot in html
    """
    jpg = get_plt_base64()
    return f'<img src="data:image/jpg;base64, {jpg}"/>'


def get_txt_html_str(txt: Union[str, List[str]], txt_type: str = 'p') -> str:
    """
    Return an html-formatted text string to display.

    Args:
        txt: the string to be printed.
        txt_type: type of text to be presented. default is h3 (header-3).

    Returns:
        string in text/html format in order to display the text in html

    """
    if isinstance(txt, list):
        txt = '<br>'.join(txt)
    return f'<{txt_type}>{txt}</{txt_type}'


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


def validate_column_list(cl) -> set:
    """Validate the object given is a list of strings or None.

    Args:
        cl: the object to validate

    Returns:
        (set): Returns a list of columns names as set object or None
    """
    var_names = 'columns & ignore_columns '

    result: set
    if cl is not None:
        if not isinstance(cl, Iterable):
            raise MLChecksValueError(var_names + 'must be an iterable')
        if len(cl) == 0:
            raise MLChecksValueError(var_names + "can't be an emptry string")
        if any((not isinstance(string, str) for string in cl)):
            raise MLChecksValueError(var_names + "must contain only items of type 'str'")
        result = set(cl)
    else:
        result = None

    return result


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
