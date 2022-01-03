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
from typing import Dict, List

import plotly.graph_objects as go
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

        ds_x = dataset.features_columns
        ds_y = dataset.label_col
        dataset_classes = dataset.classes
        multi_y = (np.array(ds_y)[:, None] == np.unique(ds_y)).astype(int)
        y_pred_prob = model.predict_proba(ds_x)

        fpr = {}
        tpr = {}
        roc_auc = {}
        for i, class_name in enumerate(dataset_classes):
            if class_name in self.excluded_classes:
                continue
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(multi_y[:, i], y_pred_prob[:, i])
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

        fig = go.Figure()
        for i, class_name in enumerate(dataset_classes):
            if class_name in self.excluded_classes:
                continue
            if len(dataset_classes) == 2:
                fig.add_trace(go.Scatter(
                    x=fpr[i],
                    y=tpr[i],
                    line_width=2,
                    name=f'auc = {roc_auc[i]:0.2f}',
                ))
                break
            else:
                fig.add_trace(go.Scatter(
                    x=fpr[i],
                    y=tpr[i],
                    line_width=2,
                    name=f'Class {class_name} (auc = {roc_auc[i]:0.2f})',
                ))
        fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    line=dict(color='#444'),
                    line_width=2, line_dash='dash',
                    showlegend=False
                ))
        fig.update_xaxes(title='False Positive Rate')
        fig.update_yaxes(title='True Positive Rate')
        if len(dataset_classes) == 2:
            fig.update_layout(title_text='Receiver operating characteristic for binary data',
                              width=900, height=500)
        else:
            fig.update_layout(title_text='Receiver operating characteristic for multi-class data',
                              width=900, height=500)

        return CheckResult(roc_auc, header='ROC Report', display=fig)

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
