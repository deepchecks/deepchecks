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
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from deepchecks.core import CheckResult
from deepchecks.utils.strings import format_number_if_not_nan

__all__ = ['create_confusion_matrix_figure', 'run_confusion_matrix_check']


def run_confusion_matrix_check(y_pred: np.ndarray, y_true: np.ndarray, with_display=True,
                               normalize_display=True) -> CheckResult:
    """Calculate confusion matrix based on predictions and true label values."""
    total_classes = sorted([str(x) for x in set(y_pred).union(set(y_true))])
    result = confusion_matrix(y_true, y_pred)

    if with_display:
        fig = create_confusion_matrix_figure(result, total_classes, normalize_display)
    else:
        fig = None

    return CheckResult(result, display=fig)


def create_confusion_matrix_figure(confusion_matrix_data: np.ndarray, classes_names: List[str],
                                   normalize_display: bool):
    """Create a confusion matrix figure.

    Parameters
    ----------
    confusion_matrix_data: np.ndarray
        2D array containing the confusion matrix.
    classes_names: List[str]
        the names of the classes to display as the axis.
    normalize_display: bool
        if True will also show normalized values by the true values.

    Returns
    -------
    plotly Figure object
        confusion matrix figure

    """
    if normalize_display:
        confusion_matrix_norm = confusion_matrix_data.astype('float') / \
                                (confusion_matrix_data.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps) * 100
        z = np.vectorize(format_number_if_not_nan)(confusion_matrix_norm)
        text_template = '%{z}%<br>(%{text})'
        color_bar_title = '% out of<br>True Values'
        plot_title = 'Percent Out of True Values (Count)'
    else:
        z = confusion_matrix_data
        color_bar_title = None
        text_template = '%{text}'
        plot_title = 'Value Count'

    fig = go.Figure(data=go.Heatmap(
        x=classes_names, y=classes_names, z=z,
        text=confusion_matrix_data, texttemplate=text_template))
    fig.data[0].colorbar.title = color_bar_title
    fig.update_layout(title=plot_title)
    fig.update_layout(height=600)
    fig.update_xaxes(title='Predicted Value', type='category', scaleanchor='y', constrain='domain')
    fig.update_yaxes(title='True value', type='category', constrain='domain', autorange='reversed')

    return fig
