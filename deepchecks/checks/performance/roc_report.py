# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The roc_report check module."""
from itertools import cycle
from typing import Dict, List
from matplotlib import pyplot as plt

import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from deepchecks import CheckResult, Dataset, SingleDatasetBaseCheck
from deepchecks.base.check import ConditionResult
from deepchecks.utils.metrics import ModelType, task_type_validation
from deepchecks.utils.strings import format_number


__all__ = ['RocReport']


class RocReport(SingleDatasetBaseCheck):
    """Calculate the AUC (Area Under Curve) for each class.

    Args:
        excluded_classes (List): List of classes to exclude from the condition.
    """

    def __init__(self, excluded_classes: List = None):
        super().__init__()
        self.excluded_classes = excluded_classes if excluded_classes else []

    def run(self, dataset: Dataset, model: BaseEstimator) -> CheckResult:
        """Run check.

        Args:
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance
            dataset: a Dataset object
        Returns:
            CheckResult: value is dictionary of class and it's auc score, displays the roc graph with each class

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance with a label
        """
        return self._roc_report(dataset, model)

    def _roc_report(self, dataset: Dataset, model):
        Dataset.validate_dataset(dataset)
        dataset.validate_label()
        task_type_validation(model, dataset, [ModelType.MULTICLASS, ModelType.BINARY])

        label = dataset.label_name
        ds_x = dataset.data[dataset.features]
        ds_y = dataset.data[label]
        multi_y = (np.array(ds_y)[:, None] == np.unique(ds_y)).astype(int)
        n_classes = ds_y.nunique()
        y_pred_prob = model.predict_proba(ds_x)

        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            if i in self.excluded_classes:
                continue
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(multi_y[:, i], y_pred_prob[:, i])
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

        def display():
            plt.cla()
            plt.clf()
            colors = cycle(['blue', 'red', 'green', 'orange', 'yellow'])
            for i, color in zip(range(n_classes), colors):
                if i in self.excluded_classes:
                    continue
                if n_classes == 2:
                    plt.plot(fpr[i], tpr[i], color=color, label=f'auc = {roc_auc[i]:0.2f}')
                    break
                else:
                    plt.plot(fpr[i], tpr[i], color=color,
                            label=f'ROC curve of class {i} (auc = {roc_auc[i]:0.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([-0.05, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            if n_classes == 2:
                plt.title('Receiver operating characteristic for binary data')
            else:
                plt.title('Receiver operating characteristic for multi-class data')
            plt.title('ROC curves')
            plt.legend(loc='lower right')

        return CheckResult(roc_auc, header='ROC Report', display=display)

    def add_condition_auc_not_less_than(self, min_auc: float = 0.7):
        """Add condition - require min allowed AUC score per class.

        Args:
            min_auc (float): Max allowed AUC score per class.

        """
        def condition(result: Dict) -> ConditionResult:
            failed_classes = []
            for item in result.items():
                class_name, score = item
                if score < min_auc:
                    failed_classes.append(f'class {class_name}: {format_number(score)}')
            if failed_classes:
                return ConditionResult(False,
                                       f'The scores that are less than the allowed AUC are: {failed_classes}')
            else:
                return ConditionResult(True)

        if self.excluded_classes:
            suffix = f' except: {self.excluded_classes}'
        else:
            suffix = ''
        return self.add_condition(f'Not less than {min_auc} AUC score for all the classes{suffix}',
                                  condition)
