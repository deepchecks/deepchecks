"""The confusion_matrix_report check module."""
import sklearn
from sklearn.base import BaseEstimator
from mlchecks.base.check import SingleDatasetBaseCheck
from mlchecks import CheckResult, Dataset

__all__ = ['ConfusionMatrixReport']


class ConfusionMatrixReport(SingleDatasetBaseCheck):
    """Return the confusion_matrix."""

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run check.

        Args:
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
            dataset: a Dataset object
        Returns:
            CheckResult: value is numpy array of the confusion matrix, displays the confusion matrix

        Raises:
            MLChecksValueError: If the object is not a Dataset instance with a label
        """
        return self._confusion_matrix_report(dataset, model)

    def _confusion_matrix_report(self, dataset: Dataset, model):
        func_name = self.__class__.__name__
        Dataset.validate_dataset(dataset, func_name)
        dataset.validate_label(func_name)

        label = dataset.label_name()
        ds_x = dataset.data[dataset.features()]
        ds_y = dataset.data[label]
        y_pred = model.predict(ds_x)

        confusion_matrix = sklearn.metrics.confusion_matrix(ds_y, y_pred)

        def display():
            sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix).plot()

        return CheckResult(confusion_matrix, check=self.__class__, display=display)

