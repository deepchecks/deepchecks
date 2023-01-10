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
from sklearn.metrics import confusion_matrix, roc_auc_score

__all__ = ['false_positive_rate_metric', 'false_negative_rate_metric', 'true_negative_rate_metric', 'roc_auc_per_class']

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.metrics import averaging_mechanism


def assert_binary_values(y):
    invalid = set(np.unique(y)) - {0, 1}
    if invalid:
        raise DeepchecksValueError(f'Expected y to be a binary matrix with only 0 and 1 but got values: {invalid}')


def assert_multi_label_shape(y):
    if not isinstance(y, np.ndarray):
        raise DeepchecksValueError(f'Expected y to be numpy array instead got: {type(y)}')
    if y.ndim != 2:
        raise DeepchecksValueError(f'Expected y to be numpy array with 2 dimensions instead got {y.ndim} dimensions.')
    assert_binary_values(y)
    # Since the metrics are not yet supporting real multi-label, make sure there isn't any row with sum larger than 1
    if y.sum(axis=1).max() > 1:
        raise DeepchecksValueError('Multi label scorers are not supported yet, the sum of a row in multi-label format '
                                   'must not be larger than 1')


def assert_single_label_shape(y):
    if not isinstance(y, np.ndarray):
        raise DeepchecksValueError(f'Expected y to be numpy array instead got: {type(y)}')
    if y.ndim != 1:
        raise DeepchecksValueError(f'Expected y to be numpy array with 1 dimension instead got {y.ndim} dimensions.')
    assert_binary_values(y)


def _false_positive_rate_per_class(y_true, y_pred, classes):  # False Positives / (False Positives + True Negatives)
    result = []
    for cls in classes:
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        matrix = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
        result.append(matrix[0, 1] / (matrix[0, 1] + matrix[1, 1]) if (matrix[0, 1] + matrix[1, 1]) > 0 else 0)
    return np.asarray(result)


def _micro_false_positive_rate(y_true, y_pred, classes):
    fp, tn = 0, 0
    for cls in classes:
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        matrix = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
        fp += matrix[0, 1]
        tn += matrix[1, 1]
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def false_positive_rate_metric(y_true, y_pred, averaging_method: str = 'per_class') -> Union[np.ndarray, float]:
    """Receive a metric which calculates false positive rate.

    The rate is calculated as: False Positives / (False Positives + True Negatives)
    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes) or (n_samples) for binary
        The labels should be passed in a sequence of sequences, with the sequence for each sample being a binary vector,
        representing the presence of the i-th label in that sample (multi-label).
    y_pred : array-like of shape (n_samples, n_classes) or (n_samples) for binary
        The predictions should be passed in a sequence of sequences, with the sequence for each sample being a binary
        vector, representing the presence of the i-th label in that sample (multi-label).
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
    # Convert multi label into single label
    if averaging_method != 'binary':
        assert_multi_label_shape(y_true)
        assert_multi_label_shape(y_pred)
        classes = range(y_true.shape[1])
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        assert_single_label_shape(y_true)
        assert_single_label_shape(y_pred)
        classes = [0, 1]

    if averaging_method == 'micro':
        return _micro_false_positive_rate(y_true, y_pred, classes)
    scores_per_class = _false_positive_rate_per_class(y_true, y_pred, classes)
    weights = [sum(y_true == cls) for cls in classes]
    return averaging_mechanism(averaging_method, scores_per_class, weights)


def _false_negative_rate_per_class(y_true, y_pred, classes):  # False Negatives / (False Negatives + True Positives)
    result = []
    for cls in classes:
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        matrix = confusion_matrix(y_true_cls, y_pred_cls)
        result.append(matrix[1, 0] / (matrix[1, 0] + matrix[0, 0]) if (matrix[1, 0] + matrix[0, 0]) > 0 else 0)
    return np.asarray(result)


def _micro_false_negative_rate(y_true, y_pred, classes):
    fn, tp = 0, 0
    for cls in classes:
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        matrix = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
        fn += matrix[1, 0]
        tp += matrix[0, 0]
    return fn / (fn + tp) if (fn + tp) > 0 else 0


def false_negative_rate_metric(y_true, y_pred, averaging_method: str = 'per_class') -> Union[np.ndarray, float]:
    """Receive a metric which calculates false negative rate.

    The rate is calculated as: False Negatives / (False Negatives + True Positives)
    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes) or (n_samples) for binary
        The labels should be passed in a sequence of sequences, with the sequence for each sample being a binary vector,
        representing the presence of the i-th label in that sample (multi-label).
    y_pred : array-like of shape (n_samples, n_classes) or (n_samples) for binary
        The predictions should be passed in a sequence of sequences, with the sequence for each sample being a binary
        vector, representing the presence of the i-th label in that sample (multi-label).
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
    # Convert multi label into single label
    # Convert multi label into single label
    if averaging_method != 'binary':
        assert_multi_label_shape(y_true)
        assert_multi_label_shape(y_pred)
        classes = range(y_true.shape[1])
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        assert_single_label_shape(y_true)
        assert_single_label_shape(y_pred)
        classes = [0, 1]

    if averaging_method == 'micro':
        return _micro_false_negative_rate(y_true, y_pred, classes)

    scores_per_class = _false_negative_rate_per_class(y_true, y_pred, classes)
    weights = [sum(y_true == cls) for cls in classes]
    return averaging_mechanism(averaging_method, scores_per_class, weights)


def _true_negative_rate_per_class(y_true, y_pred, classes):  # True Negatives / (True Negatives + False Positives)
    result = []
    for cls in classes:
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        matrix = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
        result.append(matrix[1, 1] / (matrix[1, 1] + matrix[0, 1]) if (matrix[1, 1] + matrix[0, 1]) > 0 else 0)
    return np.asarray(result)


def _micro_true_negative_rate(y_true, y_pred, classes):
    tn, fp = 0, 0
    for cls in classes:
        y_true_cls, y_pred_cls = np.asarray(y_true) == cls, np.asarray(y_pred) == cls
        matrix = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
        tn += matrix[1, 1]
        fp += matrix[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def true_negative_rate_metric(y_true, y_pred, averaging_method: str = 'per_class') -> Union[np.ndarray, float]:
    """Receive a metric which calculates true negative rate. Alternative name to the same metric is specificity.

    The rate is calculated as: True Negatives / (True Negatives + False Positives)
    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes) or (n_samples) for binary
        The labels should be passed in a sequence of sequences, with the sequence for each sample being a binary vector,
        representing the presence of the i-th label in that sample (multi-label).
    y_pred : array-like of shape (n_samples, n_classes) or (n_samples) for binary
        The predictions should be passed in a sequence of sequences, with the sequence for each sample being a binary
        vector, representing the presence of the i-th label in that sample (multi-label).
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
    # Convert multi label into single label
    if averaging_method != 'binary':
        assert_multi_label_shape(y_true)
        assert_multi_label_shape(y_pred)
        classes = range(y_true.shape[1])
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        assert_single_label_shape(y_true)
        assert_single_label_shape(y_pred)
        classes = [0, 1]

    if averaging_method == 'micro':
        return _micro_true_negative_rate(y_true, y_pred, classes)

    scores_per_class = _true_negative_rate_per_class(y_true, y_pred, classes)
    weights = [sum(y_true == cls) for cls in classes]
    return averaging_mechanism(averaging_method, scores_per_class, weights)


def roc_auc_per_class(y_true, y_pred) -> np.ndarray:
    """Receives predictions and true labels and returns the ROC AUC score for each class.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes)
        The labels should be passed in a sequence of sequences, with the sequence for each sample being a binary vector,
        representing the presence of the i-th label in that sample (multi-label).
    y_pred : array-like of shape (n_samples, n_classes)
        Predicted label probabilities.

    Returns
    -------
    roc_auc : np.ndarray
        The ROC AUC score for each class.
    """
    # Convert multi label into single label
    assert_multi_label_shape(y_true)
    classes = range(y_true.shape[1])
    y_true = np.argmax(y_true, axis=1)

    return np.array([roc_auc_score(y_true == class_name, y_pred[:, i]) for i, class_name in enumerate(classes)])
