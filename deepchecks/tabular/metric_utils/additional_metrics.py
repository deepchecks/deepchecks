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
"""Utils module containing utilities for checks working with metrics."""

from typing import Callable

import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer

__all__ = ['get_false_positive_rate_scorer_binary', 'get_false_positive_rate_scorer_per_class',
           'get_false_positive_rate_scorer_macro', 'get_false_positive_rate_scorer_weighted',
           'get_false_positive_rate_scorer_micro', 'get_false_negative_rate_scorer_binary',
           'get_false_negative_rate_scorer_per_class', 'get_false_negative_rate_scorer_macro',
           'get_false_negative_rate_scorer_micro', 'get_false_negative_rate_scorer_weighted',
           'get_true_negative_rate_scorer_binary', 'get_true_negative_rate_scorer_macro',
           'get_true_negative_rate_scorer_weighted',
           'get_true_negative_rate_scorer_micro', 'get_true_negative_rate_scorer_per_class', ]


def _false_positive_rate(y_true, y_pred):  # False Positives / (False Positives + True Negatives)
    matrix = confusion_matrix(y_true, y_pred)
    return matrix[0, 1] / (matrix[0, 1] + matrix[1, 1]) if (matrix[0, 1] + matrix[1, 1]) > 0 else 0


def _false_positive_rate_per_class(y_true, y_pred):
    result = []
    for cls in sorted(y_true.dropna().unique().tolist()):
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        result.append(_false_positive_rate(y_true_cls, y_pred_cls))
    return np.asarray(result)


def get_false_positive_rate_scorer_binary() -> Callable:
    """Get binary false positive rate scorer."""
    return make_scorer(_false_positive_rate)


def get_false_positive_rate_scorer_per_class() -> Callable:
    """Get false positive rate scorer per class."""
    return make_scorer(_false_positive_rate_per_class)


def get_false_positive_rate_scorer_macro() -> Callable:
    """Get macro false positive rate scorer."""

    def macro_false_positive_rate(y_true, y_pred):
        return np.mean(_false_positive_rate_per_class(y_true, y_pred))

    return make_scorer(macro_false_positive_rate)


def get_false_positive_rate_scorer_weighted() -> Callable:
    """Get weighted false positive rate scorer."""

    def weighted_false_positive_rate(y_true, y_pred):
        result_per_class = _false_positive_rate_per_class(y_true, y_pred)
        weights = [sum(y_true == cls) for cls in sorted(y_true.dropna().unique().tolist())]
        return np.multiply(result_per_class, weights).sum() / sum(weights)

    return make_scorer(weighted_false_positive_rate)


def get_false_positive_rate_scorer_micro() -> Callable:
    """Get micro false positive rate scorer."""

    def micro_false_positive_rate(y_true, y_pred):
        fp, tn = 0, 0
        for cls in sorted(y_true.dropna().unique().tolist()):
            y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
            matrix = confusion_matrix(y_true_cls, y_pred_cls)
            fp += matrix[0, 1]
            tn += matrix[1, 1]
        return fp / (fp + tn) if (fp + tn) > 0 else 0

    return make_scorer(micro_false_positive_rate)


def _false_negative_rate(y_true, y_pred):  # False Negatives / (False Negatives + True Positives)
    matrix = confusion_matrix(y_true, y_pred)
    return matrix[1, 0] / (matrix[1, 0] + matrix[0, 0]) if (matrix[1, 0] + matrix[0, 0]) > 0 else 0


def _false_negative_rate_per_class(y_true, y_pred):
    result = []
    for cls in sorted(y_true.dropna().unique().tolist()):
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        result.append(_false_negative_rate(y_true_cls, y_pred_cls))
    return np.asarray(result)


def get_false_negative_rate_scorer_binary() -> Callable:
    """Get binary false negative rate scorer."""
    return make_scorer(_false_negative_rate)


def get_false_negative_rate_scorer_per_class() -> Callable:
    """Get false negative rate scorer per class."""
    return make_scorer(_false_negative_rate_per_class)


def get_false_negative_rate_scorer_macro() -> Callable:
    """Get macro false negative rate scorer."""

    def macro_false_negative_rate(y_true, y_pred):
        return np.mean(_false_negative_rate_per_class(y_true, y_pred))

    return make_scorer(macro_false_negative_rate)


def get_false_negative_rate_scorer_weighted() -> Callable:
    """Get weighted false negative rate scorer."""

    def weighted_false_negative_rate(y_true, y_pred):
        result_per_class = _false_negative_rate_per_class(y_true, y_pred)
        weights = [sum(y_true == cls) for cls in sorted(y_true.dropna().unique().tolist())]
        return np.multiply(result_per_class, weights).sum() / sum(weights)

    return make_scorer(weighted_false_negative_rate)


def get_false_negative_rate_scorer_micro() -> Callable:
    """Get micro false negative rate scorer."""

    def micro_false_negative_rate(y_true, y_pred):
        fn, tp = 0, 0
        for cls in sorted(y_true.dropna().unique().tolist()):
            y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
            matrix = confusion_matrix(y_true_cls, y_pred_cls)
            fn += matrix[1, 0]
            tp += matrix[0, 0]
        return fn / (fn + tp) if (fn + tp) > 0 else 0

    return make_scorer(micro_false_negative_rate)


def _true_negative_rate(y_true, y_pred):  # True Negatives / (True Negatives + False Positives) same as specificity
    matrix = confusion_matrix(y_true, y_pred)
    return matrix[1, 1] / (matrix[1, 1] + matrix[0, 1]) if (matrix[1, 1] + matrix[0, 1]) > 0 else 0


def _true_negative_rate_per_class(y_true, y_pred):
    result = []
    for cls in sorted(y_true.dropna().unique().tolist()):
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        result.append(_true_negative_rate(y_true_cls, y_pred_cls))
    return np.asarray(result)


def get_true_negative_rate_scorer_binary() -> Callable:
    """Get binary true negative rate scorer."""
    return make_scorer(_true_negative_rate)


def get_true_negative_rate_scorer_per_class() -> Callable:
    """Get true negative rate scorer per class."""
    return make_scorer(_true_negative_rate_per_class)


def get_true_negative_rate_scorer_macro() -> Callable:
    """Get macro true negative rate scorer."""

    def macro_true_negative_rate(y_true, y_pred):
        return np.mean(_true_negative_rate_per_class(y_true, y_pred))

    return make_scorer(macro_true_negative_rate)


def get_true_negative_rate_scorer_weighted() -> Callable:
    """Get weighted true negative rate scorer."""

    def weighted_true_negative_rate(y_true, y_pred):
        result_per_class = _true_negative_rate_per_class(y_true, y_pred)
        weights = [sum(y_true == cls) for cls in sorted(y_true.dropna().unique().tolist())]
        return np.multiply(result_per_class, weights).sum() / sum(weights)

    return make_scorer(weighted_true_negative_rate)


def get_true_negative_rate_scorer_micro() -> Callable:
    """Get micro true negative rate scorer."""

    def micro_true_negative_rate(y_true, y_pred):
        tn, fp = 0, 0
        for cls in sorted(y_true.dropna().unique().tolist()):
            y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
            matrix = confusion_matrix(y_true_cls, y_pred_cls)
            tn += matrix[1, 1]
            fp += matrix[0, 1]
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    return make_scorer(micro_true_negative_rate)
