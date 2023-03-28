from typing import List

import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

from deepchecks.core import CheckResult
from deepchecks.utils.strings import format_number_if_not_nan


class ConfusionMatrixAbstract:
    """Abstract class with common methods to be inherited by
    ConfusionMatrix checks, both vision and tabular."""


    def run_check(self, y_pred: np.ndarray, y_true: np.ndarray, with_display = True,
                  normalize_data = True) -> CheckResult:
        """ Calculate confusion matrix based on predictions and true label values."""
        total_classes = sorted(set(y_pred).union(set(y_true)))
        result = confusion_matrix(y_true, y_pred)

        if with_display:
            fig = create_confusion_matrix_figure(result, total_classes, normalize_data)
        else:
            fig = None

        return CheckResult(confusion_matrix, display=fig)

def create_confusion_matrix_figure(confusion_matrix_data: np.ndarray, classes_names: List[str], normalize_data: bool):
    """Create a confusion matrix figure.

    Parameters
    ----------
    confusion_matrix_data: np.ndarray
        2D array containing the confusion matrix.
    classes_names: List[str]
        the names of the classes to display as the axis.
    normalize_data: bool
        if True will also show normalized values by the true values.

    Returns
    -------
    plotly Figure object
        confusion matrix figure

    """
    if normalize_data:
        confusion_matrix_norm = confusion_matrix_data.astype('float') / \
                                (confusion_matrix_data.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps) * 100
        z = np.vectorize(format_number_if_not_nan)(confusion_matrix_norm)
        texttemplate = '%{z}%<br>(%{text})'
        colorbar_title = '% out of<br>True Values'
        plot_title = 'Percent Out of True Values (Count)'
    else:
        z = confusion_matrix_data
        colorbar_title = None
        texttemplate = '%{text}'
        plot_title = 'Value Count'

    fig = go.Figure(data=go.Heatmap(
        x=classes_names, y=classes_names, z=z,
        text=confusion_matrix_data,texttemplate=texttemplate))
    fig.data[0].colorbar.title = colorbar_title
    fig.update_layout(title=plot_title)
    fig.update_layout(height=600)
    fig.update_xaxes(title='Predicted Value', type='category', scaleanchor='y', constrain='domain')
    fig.update_yaxes(title='True value', type='category', constrain='domain', autorange='reversed')

    return fig
