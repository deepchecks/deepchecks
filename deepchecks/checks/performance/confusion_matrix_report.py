"""The confusion_matrix_report check module."""
import sklearn
from sklearn.base import BaseEstimator

from deepchecks import CheckResult, Dataset
from deepchecks.base.check import SingleDatasetBaseCheck
from deepchecks.utils.metrics import ModelType, task_type_validation


__all__ = ['ConfusionMatrixReport']


class ConfusionMatrixReport(SingleDatasetBaseCheck):
    """Calculate the confusion matrix of the model on the given dataset."""

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run check.

        Args:
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
            dataset: a Dataset object
        Returns:
            CheckResult: value is numpy array of the confusion matrix, displays the confusion matrix

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance with a label
        """
        return self._confusion_matrix_report(dataset, model)

    def _confusion_matrix_report(self, dataset: Dataset, model):
        check_name = self.__class__.__name__
        Dataset.validate_dataset(dataset, check_name)
        dataset.validate_label(check_name)
        task_type_validation(model, dataset, [ModelType.MULTICLASS, ModelType.BINARY], check_name)

        label = dataset.label_name()
        ds_x = dataset.data[dataset.features()]
        ds_y = dataset.data[label]
        y_pred = model.predict(ds_x)

        confusion_matrix = sklearn.metrics.confusion_matrix(ds_y, y_pred)

        def display():
            sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix).plot()

        return CheckResult(confusion_matrix, check=self.__class__, display=display)

