"""The feature_importance check module."""
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from mlchecks import SingleDatasetBaseCheck, CheckResult, Dataset
from mlchecks.utils import get_plt_base64, model_type_validation
import shap


def feature_importance(dataset: Dataset, model: BaseEstimator, plot_type: str = None):
    """
    Plot feature importance for given dataset on model.

    Args:
        dataset (Dataset): A dataset object
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
        plot_type (str): type of plot that is to be displayed ('bar','beeswarm') default None
    Returns:
        CheckResult: value is the SHAP values
    """
    model_type_validation(model)

    try:
        explainer = shap.Explainer(model)
    # SHAP throws broad exception, and we want to catch it and return an empty result
    # because we don't want to affect the suite
    # pylint: disable=broad-except
    except Exception:
        return CheckResult(None,
                           {'text/html': '<p style="color:red;">model type not currently supported for SHAP calculation</p>'})

    shap_values = explainer.shap_values(dataset[dataset.features()])

    if plot_type == 'bar':
        shap.summary_plot(shap_values, dataset[dataset.features()], dataset.features(), plot_type=plot_type, show=False)
    elif plot_type == 'beeswarm' or plot_type is None:
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap.summary_plot(shap_values[1], dataset[dataset.features()], dataset.features(), show=False)
            elif plot_type is None:
                shap.summary_plot(shap_values, dataset[dataset.features()], dataset.features(), show=False)
            else:
                return CheckResult(None,
                                   {'text/html': '<p style="color:red;">Only plot_type = \'bar\''
                                                 ' is supported for multi-class models!</p>'})
        else:
            shap.summary_plot(shap_values, dataset[dataset.features()], dataset.features(), show=False)
    else:
        return CheckResult(None, {'text/html': '<p style="color:red;">unsuported plot_type</p>'})

    plot = get_plt_base64()

    # SHAP prints the plot despite show=False, here we clear the plot frame
    plt.cla()
    plt.clf()

    return CheckResult(shap_values, {'text/html': f'<img src="data:image/jpg;base64,{plot}"/>'})


class FeatureImportance(SingleDatasetBaseCheck):
    """Check class for the check function feature_importance."""

    def run(self, dataset, model=None) -> CheckResult:
        """
        Run the feature_importance check.

        Arguments:
            dataset: Dataset - The dataset object
            model: any = None - The model object

        Returns:
            the output of the feature_importance check
        """
        return feature_importance(dataset, model)
