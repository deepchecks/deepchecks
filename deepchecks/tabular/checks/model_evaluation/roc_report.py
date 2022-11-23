# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
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

import numpy as np
import plotly.graph_objects as go
import sklearn

from deepchecks.core import CheckResult, ConditionResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.dict_funcs import get_dict_entry_by_value
from deepchecks.utils.strings import format_number

__all__ = ['RocReport']


class RocReport(SingleDatasetCheck):
    """Calculate the ROC curve for each class.

    For each class, plots the ROC curve for a one vs all classification of that class based on the model's
    predicted probabilities. In addition, for each class, it marks the optimal probability threshold cut-off point
    based on Youden's index defined as sensitivity + specificity - 1.
    See https://en.wikipedia.org/wiki/Youden%27s_J_statistic for additional details.

    Parameters
    ----------
    excluded_classes : List , default: None
        List of classes to exclude from the calculation. If None, calculate one vs all ROC curve for all classes for
        multiclass and only the positive class for binary classification. If an empty list was provided,
        calculate for all classes.
    n_samples : int , default: 1_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.

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

    def __init__(self, excluded_classes: List = None,
                 n_samples: int = 1_000_000,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(**kwargs)
        self.excluded_classes = excluded_classes
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is dictionary of a class and its auc score, displays the roc graph with each class

        Raises
        ------
        DeepchecksValueError
            If the object is not a Dataset instance with a label
        """
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=self.random_state)
        context.assert_classification_task()
        if not hasattr(context.model, 'predict_proba'):
            raise DeepchecksNotSupportedError('Predicted probabilities not supplied. The roc report check '
                                              'needs the predicted probabilities to plot the ROC curve, instead'
                                              ' of only predicted classes.')

        y_pred_prob = context.model.predict_proba(dataset.features_columns)
        dataset_classes = context.model_classes

        fpr = {}
        tpr = {}
        thresholds = {}
        roc_auc = {}
        for i, class_name in enumerate(dataset_classes):
            if self.excluded_classes is not None and class_name in self.excluded_classes:
                continue
            fpr[class_name], tpr[class_name], thresholds[class_name] = \
                sklearn.metrics.roc_curve(dataset.label_col == class_name, y_pred_prob[:, i])
            roc_auc[class_name] = sklearn.metrics.auc(fpr[class_name], tpr[class_name])

        if self.excluded_classes is not None:
            classes_for_display = [x for x in dataset_classes if x not in self.excluded_classes]
        else:
            classes_for_display = [dataset_classes[1]] if context.task_type == TaskType.BINARY else dataset_classes

        if context.with_display:
            fig = go.Figure()
            for class_name in classes_for_display:
                fig.add_trace(go.Scatter(
                    x=fpr[class_name],
                    y=tpr[class_name],
                    line_width=2,
                    name=f'Class {class_name} (auc = {roc_auc[class_name]:0.2f})'
                ))
                fig.add_trace(get_cutoff_figure(tpr[class_name], fpr[class_name],
                                                thresholds[class_name], class_name))
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                line=dict(color='#444'),
                line_width=2, line_dash='dash',
                showlegend=False
            ))
            fig.update_xaxes(title='False Positive Rate')
            fig.update_yaxes(title='True Positive Rate')
            fig.update_layout(title_text='Receiver Operating Characteristic Plot', height=500)
            footnote = """The marked points are the optimal probability threshold cut-off points to predict said
            class. In plain terms, it is optimal to set the prediction rule such that if for some class the predicted
            probability is above the threshold of that class, then the prediction should be that class.
            They optimal thresholds are determined using Youden's index defined as sensitivity + specificity - 1."""
            display = [fig, footnote]
        else:
            display = None

        return CheckResult({x: roc_auc[x] for x in classes_for_display}, header='ROC Report', display=display)

    def add_condition_auc_greater_than(self, min_auc: float = 0.7):
        """Add condition - require min allowed AUC score per class.

        Parameters
        ----------
        min_auc : float , default: 0.7
            Max allowed AUC score per class.

        """

        def condition(result: Dict) -> ConditionResult:
            failed_classes = {class_name: format_number(score)
                              for class_name, score in result.items() if score <= min_auc}
            if failed_classes:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Found classes with AUC below threshold: {failed_classes}')
            else:
                class_name, score = get_dict_entry_by_value(result, value_select_fn=min)
                details = f'All classes passed, minimum AUC found is {format_number(score)} for class {class_name}'
                return ConditionResult(ConditionCategory.PASS, details)

        if self.excluded_classes:
            suffix = f' except: {self.excluded_classes}'
        else:
            suffix = ''
        return self.add_condition(f'AUC score for all the classes{suffix} is greater than {min_auc}',
                                  condition)


def get_cutoff_figure(tpr, fpr, thresholds, class_name):
    highest_youden_index = sensitivity_specificity_cutoff(tpr, fpr)
    hovertemplate = f'Class: {class_name}' + '<br>TPR: %{y:.2%}<br>FPR: %{x:.2%}' + \
                    f'<br>Optimal Threshold: {thresholds[highest_youden_index]:.3}'
    return go.Scatter(x=[fpr[highest_youden_index]], y=[tpr[highest_youden_index]], mode='markers', marker_size=15,
                      hovertemplate=hovertemplate, showlegend=False, marker={'color': 'black'})


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
