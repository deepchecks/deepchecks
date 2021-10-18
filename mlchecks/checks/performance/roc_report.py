"""The roc_report check module."""
from itertools import cycle
from matplotlib import pyplot as plt

import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from mlchecks.base.dataset import validate_dataset
from mlchecks import CheckResult, Dataset, SingleDatasetBaseCheck


__all__ = ['roc_report', 'RocReport']


def roc_report(ds: Dataset, model):
    """
    Return the AUC for each class.

    Args:
        model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
        ds: a Dataset object
    Returns:
        CheckResult: value is dictionary of class and it's auc score, displays the roc graph with each class

    Raises:
        MLChecksValueError: If the object is not a Dataset instance with a label

    """
    validate_dataset(ds, 'roc_report')
    ds.validate_label('roc_report')

    label = ds.label_name()
    ds_x = ds[ds.features()]
    ds_y = ds[label]
    multi_y = (np.array(ds_y)[:, None] == np.unique(ds_y)).astype(int)
    n_classes = ds_y.nunique()
    y_pred_prob = model.predict_proba(ds_x)

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(multi_y[:, i], y_pred_prob[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    def display():
        plt.cla()
        plt.clf()
        colors = cycle(['blue', 'red', 'green', 'orange', 'yellow'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color,
                     label=f'ROC curve of class {i} (auc = {roc_auc[i]:0.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for multi-class data')
        plt.legend(loc='lower right')

    return CheckResult(roc_auc, header='ROC Report', check=roc_report, display=display)


class RocReport(SingleDatasetBaseCheck):
    """Summarize given model parameters."""

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run roc_report check.

        Args:
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
            ds: a Dataset object
        Returns:
            CheckResult: value is dictionary of class and it's auc score, displays the roc graph with each class
        """
        return roc_report(dataset, model)
