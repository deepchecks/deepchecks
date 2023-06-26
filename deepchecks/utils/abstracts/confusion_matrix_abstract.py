# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The confusion_matrix_report check module."""
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from plotly.subplots import make_subplots

from deepchecks import ConditionCategory, ConditionResult
from deepchecks.core import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.strings import format_number_if_not_nan, format_percent

__all__ = ['create_confusion_matrix_figure', 'run_confusion_matrix_check']

MAX_WORST_CLASSES_TO_DISPLAY = 3
MAX_TOP_CLASSES_TO_DISPLAY = 3
MIN_ACCURACY_FOR_GOOD_CLASSES = 85.0

def run_confusion_matrix_check(y_pred: np.ndarray, y_true: np.ndarray, with_display=True,
                               normalize_display=True) -> CheckResult:
    """Calculate confusion matrix based on predictions and true label values."""
    total_classes = sorted([str(x) for x in set(y_pred).union(set(y_true))])
    result = confusion_matrix(y_true, y_pred)

    if with_display:
        displays = create_confusion_matrix_figure(result, total_classes)
    else:
        displays = None

    # For accessing the class names from the condition
    result = pd.DataFrame(result, index=total_classes, columns=total_classes)

    return CheckResult(result, display=displays)


def create_confusion_matrix_figure(confusion_matrix_data: np.ndarray, classes_names: List[str]):
    """Create a confusion matrix figure.

    Parameters
    ----------
    confusion_matrix_data: np.ndarray
        2D array containing the confusion matrix.
    classes_names: List[str]
        the names of the classes to display as the axis.

    Returns
    -------
    plotly Figure object
        confusion matrix figure

    """
    display = []
    confusion_matrix_norm = confusion_matrix_data.astype('float') / \
                            (confusion_matrix_data.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps) * 100

    accuracy_array = np.diag(confusion_matrix_norm).round(decimals=2)
    worst_class_indices = [index for index, acc in enumerate(accuracy_array) if acc < MIN_ACCURACY_FOR_GOOD_CLASSES]
    
    if len(accuracy_array) > MAX_WORST_CLASSES_TO_DISPLAY:
        worst_class_indices = worst_class_indices[:MAX_WORST_CLASSES_TO_DISPLAY]

    top_classes_names = []
    top_class_accuracy = []
    for index, acc in enumerate(accuracy_array):
        if index not in worst_class_indices:
            top_classes_names.append(classes_names[index])
            top_class_accuracy.append(acc)

    # Pie chart for label distribution
    total_samples_per_label = np.sum(confusion_matrix_data, axis=0)
    fig_pie = go.Figure(go.Pie(labels=classes_names, values=total_samples_per_label, title='Label Distribution<br><sup>(with accuracy and number of samples per label)</sup>',
                     textposition='inside', showlegend=False, textinfo='label+percent',
                     customdata=[[classes_names[index], acc, total_samples_per_label[index]] for index, acc in enumerate(accuracy_array)],
                     hovertemplate='Label:%{customdata[0][0]}<br>Accuracy: %{customdata[0][1]}%<br>'
                                   'Samples:%{customdata[0][2]}<extra></extra>'))
    
    display_msg = f'The overall accuracy of your model is: {round(np.sum(accuracy_array)/len(accuracy_array), 2)}%'
    display.append(display_msg)
    display.append(fig_pie)
    
    curr_col = 1
    if len(worst_class_indices) > 0:
        fig = make_subplots(rows=1, cols=min(MAX_WORST_CLASSES_TO_DISPLAY, len(worst_class_indices)),
                            specs=[[{'type': 'pie'}] * min(MAX_WORST_CLASSES_TO_DISPLAY, len(worst_class_indices))])
        for index in worst_class_indices:
            values = np.delete(confusion_matrix_data[index], index)
            n_samples_with_label = np.sum(confusion_matrix_data[index])
            total_samples = np.sum(confusion_matrix_data, axis=None)
            labels = np.delete(classes_names, index)
            fig.add_trace(go.Pie(values=values, title=f'Label Prediction Distribution: <b>{classes_names[index]}</b><br>with '
                                f'<b>{confusion_matrix_data[index][index]} ({accuracy_array[index]}%)</b> correct predictions'
                                f'<br>Having <b>{format_percent(n_samples_with_label/total_samples)}</b> of labeled data '
                                f'in the dataset', labels=labels, showlegend=False, textposition='inside', textinfo='label+percent',
                                customdata=[classes_names[index], accuracy_array[index], n_samples_with_label],
                                hovertemplate='Label:%{customdata[0]}<br>Accuracy: %{customdata[1]}%<br>'
                                   'Samples:%{customdata[2]}<extra></extra>'), row=1, col=curr_col)
            curr_col += 1
        fig.update_layout(title='Worst Performing Classes', title_x=0.5)
        display.append(fig)
    return display


def misclassified_samples_lower_than_condition(value: pd.DataFrame,
                                               misclassified_samples_threshold: float) -> ConditionResult:
    """Condition function that checks if the misclassified samples in the confusion matrix is below threshold.

    Parameters
    ----------
    value: pd.DataFrame
        Dataframe containing the confusion matrix
    misclassified_samples_threshold: float
        Ratio of samples to be used for comparison in the condition (Value should be between 0 - 1 inclusive)

    Raises
    ------
    DeepchecksValueError
        if the value of `misclassified_samples_threshold` parameter is not between 0 - 1 inclusive.

    Returns
    -------
    ConditionResult
        - ConditionCategory.PASS, if all the misclassified samples in the confusion
        matrix are less than `misclassified_samples_threshold` ratio
        - ConditionCategory.FAIL, if the misclassified samples in the confusion matrix
        are more than the `misclassified_samples_threshold` ratio
    """
    if misclassified_samples_threshold < 0 or misclassified_samples_threshold > 1:
        raise DeepchecksValueError(
           'Condition requires the parameter "misclassified_samples_threshold" '
           f'to be between 0 and 1 inclusive but got {misclassified_samples_threshold}'
        )

    # Getting the class names from the confusion matrix
    class_names = value.columns

    # Converting the confusion matrix to a numpy array for numeric indexing
    value = value.to_numpy()

    # Computing the total number of samples from the confusion matrix
    total_samples = np.sum(value)

    # Number of threshold samples based on the 'misclassified_samples_threshold' parameter
    thresh_samples = round(np.ceil(misclassified_samples_threshold * total_samples))

    # m is the number of rows in the confusion matrix and
    # n is the number of columns in the confusion matrix
    m, n = value.shape[0], value.shape[1]

    # Variables to keep track of the misclassified cells above 'thresh_samples'
    n_cells_above_thresh = 0
    max_misclassified_cell_idx = (0, 1)

    # Looping over the confusion matrix and checking only the misclassified cells
    for i in range(m):
        for j in range(n):
            # omitting the principal axis of the confusion matrix
            if i != j:
                n_samples = value[i][j]

                if n_samples > thresh_samples:
                    n_cells_above_thresh += 1

                    x, y = max_misclassified_cell_idx
                    max_misclassified_samples = value[x][y]
                    if n_samples > max_misclassified_samples:
                        max_misclassified_cell_idx = (i, j)

    # There are misclassified cells in the confusion matrix with samples more than 'thresh_samples'
    if n_cells_above_thresh > 0:
        x, y = max_misclassified_cell_idx
        max_misclassified_samples = value[x][y]
        max_misclassified_samples_ratio = max_misclassified_samples / total_samples

        details = f'Detected {n_cells_above_thresh} misclassified confusion matrix cell(s) each one ' \
                  f'containing more than {format_percent(misclassified_samples_threshold)} of the data. ' \
                  f'Largest misclassified cell ({format_percent(max_misclassified_samples_ratio)} of the data) ' \
                  f'is samples with a true value of "{class_names[x]}" and a predicted value of "{class_names[y]}".'

        return ConditionResult(ConditionCategory.FAIL, details)

    # No cell has more than 'thresh_samples' misclassified samples
    details = 'All misclassified confusion matrix cells contain less than ' \
              f'{format_percent(misclassified_samples_threshold)} of the data.'

    return ConditionResult(ConditionCategory.PASS, details)
