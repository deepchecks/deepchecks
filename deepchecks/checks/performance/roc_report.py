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
from deepchecks.utils.metrics import ModelType
from deepchecks.utils.strings import format_number


__all__ = ['RocReport']


class RocReport(SingleDatasetBaseCheck):
    """Calculate the ROC curve for each class.

    For each class plots the ROC curve, calculate AUC score and displays the optimal threshold cutoff point.

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
        dataset = Dataset.ensure_not_empty_dataset(dataset)
        ds_y = self._dataset_has_label(dataset)
        ds_x = self._dataset_has_features(dataset)
        self._verify_model_type(model, dataset, [ModelType.MULTICLASS, ModelType.BINARY])

        dataset_classes = dataset.classes
        multi_y = (np.array(ds_y)[:, None] == np.unique(ds_y)).astype(int)
        y_pred_prob = model.predict_proba(ds_x)

        fpr = {}
        tpr = {}
        thresholds = {}
        roc_auc = {}
        for i, class_name in enumerate(dataset_classes):
            if class_name in self.excluded_classes:
                continue
            fpr[class_name], tpr[class_name], thresholds[class_name] = \
                sklearn.metrics.roc_curve(multi_y[:, i], y_pred_prob[:, i])
            roc_auc[class_name] = sklearn.metrics.auc(fpr[class_name], tpr[class_name])

        fig = go.Figure()
        for class_name in dataset_classes:
            if class_name in self.excluded_classes:
                continue
            if len(dataset_classes) == 2:
                fig.add_trace(go.Scatter(
                    x=fpr[class_name],
                    y=tpr[class_name],
                    line_width=2,
                    name=f'auc = {roc_auc[class_name]:0.2f}',
                ))
                fig.add_trace(get_cutoff_figure(tpr[class_name], fpr[class_name], thresholds[class_name]))
                break
            else:
                fig.add_trace(go.Scatter(
                    x=fpr[class_name],
                    y=tpr[class_name],
                    line_width=2,
                    name=f'Class {class_name} (auc = {roc_auc[class_name]:0.2f})'
                ))
                fig.add_trace(get_cutoff_figure(tpr[class_name], fpr[class_name], thresholds[class_name], class_name))
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

        footnote = """<span style="font-size:0.8em"><i>
        The marked points are the optimal threshold cut-off points. They are determined using Youden's index defined
        as sensitivity + specificity - 1
        </i></span>"""

        return CheckResult(roc_auc, header='ROC Report', display=[fig, footnote])

    def add_condition_auc_not_less_than(self, min_auc: float = 0.7):
        """Add condition - require min allowed AUC score per class.

        Args:
            min_auc (float): Max allowed AUC score per class.

        """
        def condition(result: Dict) -> ConditionResult:
            failed_classes = {}
            for item in result.items():
                class_name, score = item
                if score < min_auc:
                    failed_classes[class_name] = format_number(score)
            if failed_classes:
                return ConditionResult(False,
                                       f'Found classes with AUC below threshold: {failed_classes}')
            else:
                return ConditionResult(True)

        if self.excluded_classes:
            suffix = f' except: {self.excluded_classes}'
        else:
            suffix = ''
        return self.add_condition(f'AUC score for all the classes{suffix} is not less than {min_auc}',
                                  condition)


def get_cutoff_figure(tpr, fpr, thresholds, class_name=None):
    index = sensitivity_specificity_cutoff(tpr, fpr)
    hovertemplate = 'TPR: %{y:.2%}<br>FPR: %{x:.2%}' + f'<br>Youden\'s Index: {thresholds[index]:.3}'
    if class_name:
        hovertemplate += f'<br>Class: {class_name}'
    return go.Scatter(x=[fpr[index]], y=[tpr[index]], mode='markers', marker_size=15,
                      hovertemplate=hovertemplate, showlegend=False)


def sensitivity_specificity_cutoff(tpr, fpr):
    """Find index of optimal cutoff point on curve.

    Cut-off is determined using Youden's index defined as sensitivity + specificity - 1.

    Parameters
    ----------
    tpr : array, shape = [n_roc_points]
        True positive rate per threshold
    fpr : array, shape = [n_roc_points]
        False positive rate per threshold

    References
    ----------
    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.

    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.

    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    """
    return np.argmax(tpr - fpr)
