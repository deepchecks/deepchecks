"""The confusion_matrix_report check module."""
import sklearn
from sklearn.base import BaseEstimator
from mlchecks.base.check import SingleDatasetBaseCheck
from mlchecks import CheckResult, Dataset

__all__ = ['confusion_matrix_report', 'ConfusionMatrixReport']


def confusion_matrix_report(dataset: Dataset, model):
    """
    Return the confusion_matrix.

    Args:
        dataset: a Dataset object
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
    Returns:
        CheckResult: value is numpy array of the confusion matrix, displays the confusion matrix

    Raises:
        MLChecksValueError: If the object is not a Dataset instance with a label

    """
    self = confusion_matrix_report
    Dataset.validate_dataset(dataset, self.__name__)
    dataset.validate_label(self.__name__)

    label = dataset.label_name()
    ds_x = dataset[dataset.features()]
    ds_y = dataset[label]
    y_pred = model.predict(ds_x)

    confusion_matrix = sklearn.metrics.confusion_matrix(ds_y, y_pred)

    def display():
        sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix).plot()

    return CheckResult(confusion_matrix, check=self, display=display)


class ConfusionMatrixReport(SingleDatasetBaseCheck):
    """Summarize given model parameters."""

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run confusion_matrix_report check.

        Args:
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
            ds: a Dataset object
        Returns:
            CheckResult: value is numpy array of the confusion matrix, displays the confusion matrix
        """
        return confusion_matrix_report(dataset, model)
