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
"""Module containing utils for semantic segmentation metrics utils."""
from typing import Tuple

import numpy as np


def format_segmentation_masks(y_true: np.ndarray, y_pred: np.ndarray, threshold):
    """Bring the ground truth and the prediction masks to the same format (C, W, H) with values 1.0 or 0.0."""
    pred_onehot = np.where(y_pred > threshold, 1.0, 0.0)
    label_onehot = np.zeros_like(pred_onehot)
    for channel in range(pred_onehot.shape[0]):
        label_onehot[channel] = y_true == channel
    return label_onehot, pred_onehot


def segmentation_counts_per_class(y_true_onehot: np.ndarray, y_pred_onehot: np.ndarray):
    """Compute the ground truth, predicted and intersection areas per class for segmentation metrics."""
    tp_onehot = np.logical_and(y_true_onehot, y_pred_onehot)
    tp_count_per_class = np.asarray([np.sum(tp_onehot[channel]) for channel in range(tp_onehot.shape[0])])
    y_true_count_per_class = np.asarray([np.sum(y_true_onehot[channel]) for channel in range(tp_onehot.shape[0])])
    pred_count_per_class = np.asarray([np.sum(y_pred_onehot[channel]) for channel in range(tp_onehot.shape[0])])
    return tp_count_per_class, y_true_count_per_class, pred_count_per_class


def segmentation_counts_micro(y_true_onehot: np.ndarray, y_pred_onehot: np.ndarray) -> Tuple[int, int, int]:
    """Compute the micro averaged ground truth, predicted and intersection areas for segmentation metrics."""
    tp_onehot = np.logical_and(y_true_onehot, y_pred_onehot)
    return np.sum(tp_onehot), np.sum(y_true_onehot), np.sum(y_pred_onehot)
