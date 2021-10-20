"""The classification_report check module."""
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck


__all__ = ['classification_report', 'ClassificationReport']


def classification_report(dataset: Dataset, model):
    """
    Return the sklearn classification_report in dict format.

    Args:
        dataset (Dataset): a Dataset object
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance

    Returns:
        CheckResult:
            value is dictionary in format
                {<target>: , ['precision': <score>, 'recall': <score>, 'f_score': <score>, 'support': <score>]}

    Raises:
        MLChecksValueError: If the object is not a Dataset instance with a label

    """
    self = classification_report
    Dataset.validate_dataset(dataset, self.__name__)
    dataset.validate_label(self.__name__)

    label = dataset.label_name()
    ds_x = dataset[dataset.features()]
    ds_y = dataset[label]
    y_pred = model.predict(ds_x)

    macro_performance = pd.DataFrame(sklearn.metrics.precision_recall_fscore_support(ds_y, y_pred))
    macro_performance.index = ['precision', 'recall', 'f_score', 'support']

    return CheckResult(macro_performance.to_dict(), check=self, display=macro_performance)


class ClassificationReport(SingleDatasetBaseCheck):
    """Summarize given model parameters."""

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run classification_report check.

        Args:
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
            ds: a Dataset object
        Returns:
            CheckResult:
                value is dictionary in format
                    {<target>: , ['precision': <score>, 'recall': <score>, 'f_score': <score>, 'support': <score>]}
        """
        return classification_report(dataset, model)
