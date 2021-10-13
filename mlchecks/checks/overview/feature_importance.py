"""The feature_importance check module."""
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from mlchecks import SingleDatasetBaseCheck, CheckResult, Dataset, validate_dataset
from mlchecks.utils import model_type_validation, get_plt_html_str, MLChecksValueError, model_dataset_shape_validation
from mlchecks.display import format_check_display

import shap


def feature_importance(dataset: Dataset, model: BaseEstimator, plot_type: str = None):
    """Plot SHAP feature importance for given dataset on model.

    Args:
        dataset (Dataset): A dataset object
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
        plot_type (str): type of plot that is to be displayed ('bar','beeswarm', None) default None
    Returns:
        CheckResult: value is the SHAP values
    """
    check_name = 'feature_importance'
    model_type_validation(model)
    validate_dataset(dataset, check_name)
    dataset.validate_label(check_name)
    model_dataset_shape_validation(model, dataset)


    try:
        explainer = shap.Explainer(model)
    # SHAP throws broad exception, and we want to catch it and return an empty result
    # because we don't want to affect the suite
    # pylint: disable=broad-except
    except Exception:
        return CheckResult(None,
                           {'text/html': format_check_display('Feature Importance', feature_importance,
                                                              '<p style="color:red;">Model type not currently supported'
                                                              ' for SHAP calculation</p>')})

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
                raise MLChecksValueError('Only plot_type = \'bar\' is supported for multi-class models</p>')
        else:
            shap.summary_plot(shap_values, dataset[dataset.features()], dataset.features(), show=False)
    else:
        raise MLChecksValueError(f'plot_type=\'{plot_type}\' currently not supported. Use \'beeswarm\' or \'bar\'')

    plot = get_plt_html_str()

    return CheckResult(shap_values, {'text/html': format_check_display('Feature Importance', feature_importance, plot)})


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
        return feature_importance(dataset, model, self.params.get('plot_type'))
