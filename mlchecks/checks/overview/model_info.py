from mlchecks import ModelOnlyBaseCheck, CheckResult
import pandas as pd
from sklearn.base import BaseEstimator


def model_info(model: BaseEstimator):
    _type = type(model).__name__
    model_param_df = pd.DataFrame.from_dict(model.get_params(), orient='index', columns=['value'])
    model_param_df.index.name = 'parameter'
    model_param_df.reset_index(inplace=True)

    html = f'<h2>{_type}</h2><br>{model_param_df.to_html(index=False)}'
    value = {'type': _type, 'params': model.get_params()}

    return CheckResult(value, display={'text/html': html})


class ModelInfo(ModelOnlyBaseCheck):
    def run(self, model: BaseEstimator) -> CheckResult:
        return model_info(model)

