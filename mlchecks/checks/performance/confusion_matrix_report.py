"""The confusion_matrix_report check module."""
import sklearn
from sklearn.base import BaseEstimator
from mlchecks.base.check import SingleDatasetBaseCheck
from mlchecks.base.dataset import validate_dataset

from mlchecks.utils import MLChecksValueError, get_plt_html_str
from mlchecks import CheckResult, Dataset

__all__ = ['confusion_matrix_report', 'ConfusionMatrixReport']


def confusion_matrix_report(ds: Dataset, model):
    """
    Return the confusion_matrix


    Args:
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
        ds: a Dataset object
    Returns:
        CheckResult: value is numpy array of the confusion matrix, displays the confusion matrix

    Raises:
        MLChecksValueError: If the object is not a Dataset instance with a label

    """
    validate_dataset(ds, 'confusion_matrix_report')
    ds.validate_label('confusion_matrix_report')

    label = ds.label_name()
    ds_x = ds[ds.features()]
    ds_y = ds[label]
    y_pred = model.predict(ds_x)

    confusion_matrix = sklearn.metrics.confusion_matrix(ds_y, y_pred)
    sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix).plot()
    
    return CheckResult(confusion_matrix, display={'text/html': get_plt_html_str()})


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
