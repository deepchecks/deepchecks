"""The feature_importance check module."""
from sklearn.base import BaseEstimator
from mlchecks import SingleDatasetBaseCheck, CheckResult, Dataset
from mlchecks.utils import model_type_validation, MLChecksValueError

import shap


class FeatureImportance(SingleDatasetBaseCheck):
    """Check class for the check function feature_importance."""

    def __init__(self, plot_type: str = None, **params):
        """Initialize the FeatureImportance check.

        Args:
            plot_type (str): type of plot that is to be displayed ('bar','beeswarm', None) default None
        """
        super().__init__(**params)
        self.plot_type = plot_type

    def run(self, dataset, model=None) -> CheckResult:
        """
        Plot SHAP feature importance for given dataset on model.

        Arguments:
            dataset: Dataset - The dataset object
            model: any = None - The model object

        Returns:
            CheckResult: value is the SHAP values
        """
        return self._feature_importance(dataset, model)

    def _feature_importance(self, dataset: Dataset, model: BaseEstimator):
        """Plot SHAP feature importance for given dataset on model.

        Args:
            dataset (Dataset): A dataset object
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
        Returns:
            CheckResult: value is the SHAP values
        """
        func_name = self._feature_importance.__name__
        Dataset.validate_dataset(dataset, func_name)
        dataset.validate_label(func_name)
        model_type_validation(model)
        dataset.validate_model(model)

        try:
            explainer = shap.Explainer(model)
        # SHAP throws broad exception, and we want to catch it and return an empty result
        # because we don't want to affect the suite
        # pylint: disable=broad-except
        except Exception:
            display = '<p style="color:red;">Model type not currently supported for SHAP calculation</p>'
            return CheckResult(None, header='Feature Importance', check=self._feature_importance, display=display)

        shap_values = explainer.shap_values(dataset.data[dataset.features()])

        def plot():
            if self.plot_type == 'bar':
                shap.summary_plot(shap_values, dataset.data[dataset.features()], dataset.features(),
                                  plot_type=self.plot_type,
                                  show=False)
            elif self.plot_type == 'beeswarm' or self.plot_type is None:
                if isinstance(shap_values, list):
                    if len(shap_values) == 2:
                        shap.summary_plot(shap_values[1], dataset.data[dataset.features()], dataset.features(),
                                          show=False)
                    elif self.plot_type is None:
                        shap.summary_plot(shap_values, dataset.data[dataset.features()], dataset.features(), show=False)
                    else:
                        raise MLChecksValueError('Only plot_type = \'bar\' is supported for multi-class models</p>')
                else:
                    shap.summary_plot(shap_values, dataset.data[dataset.features()], dataset.features(), show=False)
            else:
                raise MLChecksValueError(
                    f'plot_type=\'{self.plot_type}\' currently not supported. Use \'beeswarm\' or \'bar\'')

        return CheckResult(shap_values, display=plot, check=self.run)
