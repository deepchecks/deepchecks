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

import torch


def format_segmentation_masks(y_true: torch.Tensor, y_pred: torch.Tensor, threshold):
    """Bring the ground truth and the prediction masks to the same format (C, W, H) with values 1.0 or 0.0."""
    y_true = y_true.to(y_pred.device)
    pred_onehot = torch.where(y_pred > threshold, 1.0, 0.0)
    y_gt_i = y_true.clone().unsqueeze(0).type(torch.int64)
    gt_onehot = torch.zeros_like(pred_onehot)
    gt_onehot.scatter_(0, y_gt_i, 1.0)
    return gt_onehot, pred_onehot


def segmentation_counts_per_class(y_true_onehot: torch.Tensor, y_pred_onehot: torch.Tensor):
    """Compute the ground truth, predicted and intersection areas per class for segmentation metrics."""
    tp_onehot = torch.logical_and(y_true_onehot, y_pred_onehot)
    tp_count_per_class = torch.sum(tp_onehot, dim=[1, 2])
    gt_count_per_class = torch.sum(y_true_onehot, dim=[1, 2])
    pred_count_per_class = torch.sum(y_pred_onehot, dim=[1, 2])
    return tp_count_per_class, gt_count_per_class, pred_count_per_class


def segmentation_counts_micro(y_true_onehot: torch.Tensor, y_pred_onehot: torch.Tensor):
    """Compute the micro averaged ground truth, predicted and intersection areas for segmentation metrics."""
    tp_onehot = torch.logical_and(y_true_onehot, y_pred_onehot)
    tp_count_per_class = torch.sum(tp_onehot).unsqueeze_(0)
    gt_count_per_class = torch.sum(y_true_onehot).unsqueeze_(0)
    pred_count_per_class = torch.sum(y_pred_onehot).unsqueeze_(0)
    return tp_count_per_class, gt_count_per_class, pred_count_per_class
