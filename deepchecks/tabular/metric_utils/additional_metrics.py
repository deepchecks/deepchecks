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
"""Utils module containing additional metrics that can be used via scorers."""

from typing import Union

import numpy as np
from sklearn.metrics import confusion_matrix

__all__ = ['get_false_positive_rate_scorer', 'get_false_negative_rate_scorer', 'get_true_negative_rate_scorer']


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


def get_false_positive_rate_scorer(y_true, y_pred, averaging_method: str = 'per_class') -> Union[np.ndarray, float]:
    """Get false positive rate scorer."""
    if averaging_method == 'micro':
        return _micro_false_positive_rate(y_true, y_pred)

    scores_per_class = _false_positive_rate_per_class(y_true, y_pred)
    return _averaging_mechanism(averaging_method, scores_per_class, y_true)


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


def get_false_negative_rate_scorer(y_true, y_pred, averaging_method: str = 'per_class') -> Union[np.ndarray, float]:
    """Get false negative rate scorer."""
    if averaging_method == 'micro':
        return _micro_false_negative_rate(y_true, y_pred)

    scores_per_class = _false_negative_rate_per_class(y_true, y_pred)
    return _averaging_mechanism(averaging_method, scores_per_class, y_true)


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


def get_true_negative_rate_scorer(y_true, y_pred, averaging_method: str = 'per_class') -> Union[np.ndarray, float]:
    """Get false negative rate scorer, same as specificity."""
    if averaging_method == 'micro':
        return _micro_true_negative_rate(y_true, y_pred)

    scores_per_class = _true_negative_rate_per_class(y_true, y_pred)
    return _averaging_mechanism(averaging_method, scores_per_class, y_true)


def _averaging_mechanism(averaging_method, scores_per_class, y_true):
    if averaging_method == 'binary':
        return scores_per_class[1]
    elif averaging_method == 'per_class':
        return np.asarray(scores_per_class)
    elif averaging_method == 'macro':
        return np.mean(scores_per_class)
    elif averaging_method == 'weighted':
        weights = [sum(y_true == cls) for cls in sorted(y_true.dropna().unique().tolist())]
        return np.multiply(scores_per_class, weights).sum() / sum(weights)
    else:
        raise ValueError(f'Unknown averaging {averaging_method}')
