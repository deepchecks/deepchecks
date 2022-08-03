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
"""Utils module containing additional classification metrics that can be used via scorers."""

from typing import Union

import numpy as np
from sklearn.metrics import confusion_matrix

__all__ = ['false_positive_rate_metric', 'false_negative_rate_metric', 'true_negative_rate_metric']

from deepchecks.utils.metrics import averaging_mechanism


def _false_positive_rate_per_class(y_true, y_pred):  # False Positives / (False Positives + True Negatives)
    result = []
    for cls in sorted(y_true.dropna().unique().tolist()):
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        matrix = confusion_matrix(y_true_cls, y_pred_cls)
        result.append(matrix[0, 1] / (matrix[0, 1] + matrix[1, 1]) if (matrix[0, 1] + matrix[1, 1]) > 0 else 0)
    return np.asarray(result)


def _micro_false_positive_rate(y_true, y_pred):
    fp, tn = 0, 0
    for cls in sorted(y_true.dropna().unique().tolist()):
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        matrix = confusion_matrix(y_true_cls, y_pred_cls)
        fp += matrix[0, 1]
        tn += matrix[1, 1]
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def false_positive_rate_metric(y_true, y_pred, averaging_method: str = 'per_class') -> Union[np.ndarray, float]:
    """Receive a metric which calculates false positive rate.

    The rate is calculated as: False Positives / (False Positives + True Negatives)
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    averaging_method : str, default: 'per_class'
        Determines which averaging method to apply, possible values are:
        'per_class': Return a np array with the scores for each class (sorted by class name).
        'binary': Returns the score for the positive class. Should be used only in binary classification cases.
        'micro': Returns the micro-averaged score.
        'macro': Returns the mean of scores per class.
        'weighted': Returns a weighted mean of scores based of the class size in y_true.
    Returns
    -------
    score : Union[np.ndarray, float]
        The score for the given metric.
    """
    if averaging_method == 'micro':
        return _micro_false_positive_rate(y_true, y_pred)

    scores_per_class = _false_positive_rate_per_class(y_true, y_pred)
    weights = [sum(y_true == cls) for cls in sorted(y_true.dropna().unique().tolist())]
    return averaging_mechanism(averaging_method, scores_per_class, weights)


def _false_negative_rate_per_class(y_true, y_pred):  # False Negatives / (False Negatives + True Positives)
    result = []
    for cls in sorted(y_true.dropna().unique().tolist()):
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        matrix = confusion_matrix(y_true_cls, y_pred_cls)
        result.append(matrix[1, 0] / (matrix[1, 0] + matrix[0, 0]) if (matrix[1, 0] + matrix[0, 0]) > 0 else 0)
    return np.asarray(result)


def _micro_false_negative_rate(y_true, y_pred):
    fn, tp = 0, 0
    for cls in sorted(y_true.dropna().unique().tolist()):
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        matrix = confusion_matrix(y_true_cls, y_pred_cls)
        fn += matrix[1, 0]
        tp += matrix[0, 0]
    return fn / (fn + tp) if (fn + tp) > 0 else 0


def false_negative_rate_metric(y_true, y_pred, averaging_method: str = 'per_class') -> Union[np.ndarray, float]:
    """Receive a metric which calculates false negative rate.

    The rate is calculated as: False Negatives / (False Negatives + True Positives)
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    averaging_method : str, default: 'per_class'
        Determines which averaging method to apply, possible values are:
        'per_class': Return a np array with the scores for each class (sorted by class name).
        'binary': Returns the score for the positive class. Should be used only in binary classification cases.
        'micro': Returns the micro-averaged score.
        'macro': Returns the mean of scores per class.
        'weighted': Returns a weighted mean of scores based of the class size in y_true.
    Returns
    -------
    score : Union[np.ndarray, float]
        The score for the given metric.
    """
    if averaging_method == 'micro':
        return _micro_false_negative_rate(y_true, y_pred)

    scores_per_class = _false_negative_rate_per_class(y_true, y_pred)
    weights = [sum(y_true == cls) for cls in sorted(y_true.dropna().unique().tolist())]
    return averaging_mechanism(averaging_method, scores_per_class, weights)


def _true_negative_rate_per_class(y_true, y_pred):  # True Negatives / (True Negatives + False Positives)
    result = []
    for cls in sorted(y_true.dropna().unique().tolist()):
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        matrix = confusion_matrix(y_true_cls, y_pred_cls)
        result.append(matrix[1, 1] / (matrix[1, 1] + matrix[0, 1]) if (matrix[1, 1] + matrix[0, 1]) > 0 else 0)
    return np.asarray(result)


def _micro_true_negative_rate(y_true, y_pred):
    tn, fp = 0, 0
    for cls in sorted(y_true.dropna().unique().tolist()):
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        matrix = confusion_matrix(y_true_cls, y_pred_cls)
        tn += matrix[1, 1]
        fp += matrix[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def true_negative_rate_metric(y_true, y_pred, averaging_method: str = 'per_class') -> Union[np.ndarray, float]:
    """Receive a metric which calculates true negative rate. Alternative name to the same metric is specificity.

    The rate is calculated as: True Negatives / (True Negatives + False Positives)
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    averaging_method : str, default: 'per_class'
        Determines which averaging method to apply, possible values are:
        'per_class': Return a np array with the scores for each class (sorted by class name).
        'binary': Returns the score for the positive class. Should be used only in binary classification cases.
        'micro': Returns the micro-averaged score.
        'macro': Returns the mean of scores per class.
        'weighted': Returns a weighted mean of scores based of the class size in y_true.
    Returns
    -------
    score : Union[np.ndarray, float]
        The score for the given metric.
    """
    if averaging_method == 'micro':
        return _micro_true_negative_rate(y_true, y_pred)

    scores_per_class = _true_negative_rate_per_class(y_true, y_pred)
    weights = [sum(y_true == cls) for cls in sorted(y_true.dropna().unique().tolist())]
    return averaging_mechanism(averaging_method, scores_per_class, weights)
