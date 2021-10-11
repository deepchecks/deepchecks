from mlchecks import ModelOnlyBaseCheck, CheckResult
import pandas as pd
from sklearn.base import BaseEstimator

from mlchecks.utils import SUPPORTED_BASE_MODELS, MLChecksException


def model_info(model: BaseEstimator):
    """Summarize given model parameters

    Args:
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

    Returns:
        CheckResult: value is dictionary in format {type: <model_type>, params: <model_params_dict>}
    """
    if not any([isinstance(model, base) for base in SUPPORTED_BASE_MODELS]):
        raise MLChecksException(f'Model must inherit from one of supported models: {SUPPORTED_BASE_MODELS}')

    _type = type(model).__name__
    model_param_df = pd.DataFrame.from_dict(model.get_params(), orient='index', columns=['value'])
    model_param_df.index.name = 'parameter'
    model_param_df.reset_index(inplace=True)

    html = f'<h2>{_type}</h2><br>{model_param_df.to_html(index=False)}'
    value = {'type': _type, 'params': model.get_params()}

    return CheckResult(value, display={'text/html': html})


class ModelInfo(ModelOnlyBaseCheck):
    """
    Class that wraps the usage of `model_info` to be used in `Suite`
    """
    def run(self, model: BaseEstimator) -> CheckResult:
        return model_info(model)

