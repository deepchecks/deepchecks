from mlchecks import Model, Check, CheckResult
import pandas as pd


def model_info(model: Model):
    _type = type(model.model_obj).__name__
    model_param_df = pd.DataFrame.from_dict(model.model_obj.get_params(), orient='index', columns=['value'])
    model_param_df.index.name = 'parameter'
    model_param_df.reset_index(inplace=True)

    html = f'<h2>{_type}</h2><br>{model_param_df.to_html(index=False)}'
    return CheckResult(None, display={'text/html': html})


class ModelInfo(Check):
    def run(self, model=None, train_data=None, validation_data=None) -> CheckResult:
        return model_info(model)

