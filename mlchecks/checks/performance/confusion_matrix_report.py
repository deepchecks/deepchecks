"""The confusion_matrix_report check module."""
import sklearn
from sklearn.base import BaseEstimator
from mlchecks.base.check import SingleDatasetBaseCheck
from mlchecks.base.dataset import validate_dataset

from mlchecks.utils import MLChecksValueError, get_plt_html_str
from mlchecks import CheckResult, Dataset

__all__ = ['confusion_matrix_report', 'ConfusionMatrixReport']


def confusion_matrix_report(ds: Dataset, model):
    validate_dataset(ds, 'confusion_matrix_report')
    ds.validate_label('confusion_matrix_report')

    label = ds.label_name()
    res = dict()
    ds_x = ds[ds.features()]
    ds_y = ds[label]
    y_pred = model.predict(ds_x)

    confusion_matrix = sklearn.metrics.confusion_matrix(ds_y, y_pred)
    sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix).plot()
    
    return CheckResult(res, display={'text/html': get_plt_html_str})


class ConfusionMatrixReport(SingleDatasetBaseCheck):
    """Summarize given model parameters."""

    def run(self, model: BaseEstimator, ds: Dataset) -> CheckResult:
        """Run confusion_matrix_report check.

        Args:
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
            ds: a Dataset object
        Returns:
            CheckResult: value is numpy array of the confusion matrix, displays the confusion matrix
        """
        if not ds.label_col():
            raise MLChecksValueError("Dataset doesn't have label column configured")
        return confusion_matrix_report(model)
