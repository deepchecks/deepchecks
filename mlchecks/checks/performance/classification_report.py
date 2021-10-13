"""The classification_report check module."""
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from mlchecks.base.check import SingleDatasetBaseCheck
from mlchecks.base.dataset import validate_dataset
from mlchecks.display import format_check_display

from mlchecks import CheckResult, Dataset


__all__ = ['classification_report', 'ClassificationReport']


def classification_report(ds: Dataset, model):
    """
    Return the sklearn classification_report in dict format.

    Args:
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
        ds: a Dataset object
    Returns:
        CheckResult: value is dictionary in format {<target>: , ['precision': <score>, 'recall': <score>, 'f_score': <score>, 'support': <score>]}

    Raises:
        MLChecksValueError: If the object is not a Dataset instance with a label

    """
    validate_dataset(ds, 'classification_report')
    ds.validate_label('classification_report')

    label = ds.label_name()
    ds_x = ds[ds.features()]
    ds_y = ds[label]
    y_pred = model.predict(ds_x)

    macro_performance = pd.DataFrame(sklearn.metrics.precision_recall_fscore_support(ds_y, y_pred))
    macro_performance.index = ['precision', 'recall', 'f_score', 'support']

    return CheckResult(macro_performance.to_dict(), display={'text/html': format_check_display('Classification Report', classification_report, macro_performance.to_html())})

class ClassificationReport(SingleDatasetBaseCheck):
    """Summarize given model parameters."""

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run classification_report check.

        Args:
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
            ds: a Dataset object
        Returns:
            CheckResult: value is dictionary in format {<target>: , ['precision': <score>, 'recall': <score>, 'f_score': <score>, 'support': <score>]}
        """
        return classification_report(dataset, model)
