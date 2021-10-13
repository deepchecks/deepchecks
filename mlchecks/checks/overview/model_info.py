"""Module contains model_info check."""
from mlchecks import ModelOnlyBaseCheck, CheckResult
from mlchecks.display import format_check_display
from mlchecks.utils import model_type_validation
import pandas as pd
from sklearn.base import BaseEstimator

__all__ = ['model_info', 'ModelInfo']


def model_info(model: BaseEstimator):
    """
    Summarize given model parameters.

    Args:
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

    Returns:
        CheckResult: value is dictionary in format {type: <model_type>, params: <model_params_dict>}
    """
    model_type_validation(model)
    model_type = type(model).__name__
    model_param_df = pd.DataFrame.from_dict(model.get_params(), orient='index', columns=['value'])
    model_param_df.index.name = 'parameter'
    model_param_df.reset_index(inplace=True)

    value = {'type': model_type, 'params': model.get_params()}
    formatted_html = format_check_display('Model Info', model_info, model_param_df.to_html(index=False))

    return CheckResult(value, display={'text/html': formatted_html})


class ModelInfo(ModelOnlyBaseCheck):
    """Summarize given model parameters."""

    def run(self, model: BaseEstimator) -> CheckResult:
        """Run model_info check.

        Args:
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult: value is dictionary in format {type: <model_type>, params: <model_params_dict>}
        """
        return model_info(model)

