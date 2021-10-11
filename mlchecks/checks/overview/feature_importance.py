from sklearn.base import BaseEstimator
from mlchecks import SingleDatasetBaseCheck, CheckResult, Dataset
import shap

def feature_importance(dataset: Dataset, model: BaseEstimator, plot_type="bar"):
    """Plot feature importance for given dataset on model

    Args:
        dataset (Dataset): mlChecks dataset to process
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
    """

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(dataset[dataset.features()])
    except:
        return CheckResult(None, {'text/html': '<p style="color:red;">model type not supported for SHAP calculation</p>'})

    if plot_type == "beeswarm":
        if type(shap_values) is list:
            if len(shap_values) == 2:
                plot = shap.summary_plot(shap_values[1], dataset[dataset.features()], dataset.features(), plot_type=plot_type)
            else:
                return CheckResult(None,  {'text/html': '<p style="color:red;">Only plot_type = \'bar\' is supported for multi-class models!</p>'})
        else:
            plot = shap.summary_plot(shap_values, dataset[dataset.features()], dataset.features(),
                                     plot_type=plot_type)

    elif plot_type == "bar":
        plot = shap.summary_plot(shap_values, dataset[dataset.features()], dataset.features(), plot_type=plot_type)
    else:
        return CheckResult(None, {'text/html': '<p style="color:red;">unsuported ploty_type</p>'})

    return CheckResult(shap_values, {"html": "todo"})


class FeatureImportance(SingleDatasetBaseCheck):
    """
    Check class for the check function feature_importance
    """
    def run(self, dataset, model=None) -> CheckResult:
        return feature_importance(dataset, model)
