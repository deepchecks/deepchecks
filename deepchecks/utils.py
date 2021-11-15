"""Utils module containing useful global functions."""
import re
import subprocess
from typing import Any
import sklearn
from IPython import get_ipython

__all__ = ['DeepchecksValueError', 'model_type_validation', 'is_notebook', 'is_widgets_enabled']

# Need to test only once if running in notebook so cache result
_is_notebook: bool = None
_is_widgets_enabled: bool = None


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
    global _is_notebook
    if _is_notebook is None:
        try:
            shell = get_ipython().__class__.__name__
            _is_notebook = shell == 'ZMQInteractiveShell'
        except NameError:
            _is_notebook = False      # Probably standard Python interpreter
    return _is_notebook


def is_widgets_enabled():
    """Check if we're running in jupyter and having jupyter widgets extension enabled."""
    global _is_widgets_enabled
    if _is_widgets_enabled is None:
        if not is_notebook():
            _is_widgets_enabled = False
        else:
            # Test if widgets extension are in list
            try:
                # The same widget can appear multiple times from different config locations, than if there are both
                # disabled and enabled, regard it as disabled
                output = subprocess.getoutput('jupyter nbextension list').split('\n')
                disabled_regex = re.compile(r'\s*(jupyter-js-widgets/extension).*(disabled).*')
                enabled_regex = re.compile(r'\s*(jupyter-js-widgets/extension).*(disabled).*')
                found_disabled = any((disabled_regex.match(s) for s in output))
                found_enabled = any((enabled_regex.match(s) for s in output))
                return not found_disabled and found_enabled
            # pylint: disable=bare-except
            except:
                _is_widgets_enabled = False

    return _is_widgets_enabled
