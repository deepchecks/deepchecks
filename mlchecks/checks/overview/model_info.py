"""Module contains model_info check."""
from mlchecks import ModelOnlyBaseCheck, CheckResult
import pandas as pd
from sklearn.base import BaseEstimator
from mlchecks.utils import model_type_validation

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
    _type = type(model).__name__
    model_param_df = pd.DataFrame.from_dict(model.get_params(), orient='index', columns=['value'])
    model_param_df.index.name = 'parameter'
    model_param_df.reset_index(inplace=True)

    html = f'<h2>{_type}</h2><br>{model_param_df.to_html(index=False)}'
    value = {'type': _type, 'params': model.get_params()}

    return CheckResult(value, display={'text/html': html})


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

