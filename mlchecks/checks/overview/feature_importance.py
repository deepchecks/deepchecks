"""The feature_importance check module."""
from sklearn.base import BaseEstimator
from mlchecks import SingleDatasetBaseCheck, CheckResult, Dataset
from mlchecks.utils import model_type_validation, MLChecksValueError, model_dataset_shape_validation

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
    self = feature_importance
    Dataset.validate_dataset(dataset, self.__name__)
    dataset.validate_label(self.__name__)
    model_type_validation(model)
    model_dataset_shape_validation(model, dataset)
    print('run!')

    try:
        explainer = shap.Explainer(model)
    # SHAP throws broad exception, and we want to catch it and return an empty result
    # because we don't want to affect the suite
    # pylint: disable=broad-except
    except Exception:
        display = '<p style="color:red;">Model type not currently supported for SHAP calculation</p>'
        return CheckResult(None, header='Feature Importance', check=feature_importance, display=display)

    shap_values = explainer.shap_values(dataset.data[dataset.features()])

    def plot():
        if plot_type == 'bar':
            shap.summary_plot(shap_values, dataset.data[dataset.features()], dataset.features(), plot_type=plot_type,
                              show=False)
        elif plot_type == 'beeswarm' or plot_type is None:
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap.summary_plot(shap_values[1], dataset.data[dataset.features()], dataset.features(), show=False)
                elif plot_type is None:
                    shap.summary_plot(shap_values, dataset.data[dataset.features()], dataset.features(), show=False)
                else:
                    raise MLChecksValueError('Only plot_type = \'bar\' is supported for multi-class models</p>')
            else:
                shap.summary_plot(shap_values, dataset.data[dataset.features()], dataset.features(), show=False)
        else:
            raise MLChecksValueError(f'plot_type=\'{plot_type}\' currently not supported. Use \'beeswarm\' or \'bar\'')

    return CheckResult(shap_values, display=plot, check=self)


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
