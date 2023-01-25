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
"""Module for calculating semantic segmentation metrics."""
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
from ignite.metrics import Metric

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.metrics_utils.semantic_segmentation_metric_utils import (format_segmentation_masks,
                                                                                segmentation_counts_micro,
                                                                                segmentation_counts_per_class)


class MeanDice(Metric):
    """Metric that calculates the mean Dice metric for each class.

    See more: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Parameters
    ----------
    threshold: float, default: 0.5
        prediction value per pixel above which the pixel is considered True.
    smooth: float, default: 1e-3
        smoothing factor to prevent division by zero when the mask is empty.
    average: str, default: none
        The method for averaging over the classes. If none, returns the result per class.
    """

    def __init__(self, *args, threshold: float = 0.5, smooth=1e-3, average: str = 'none', **kwargs):
        super().__init__(*args, **kwargs)
        self._evals = defaultdict(lambda: {'dice': 0, 'count': 0})
        self.threshold = threshold
        self.smooth = smooth
        if average in ['none', 'micro', 'macro']:
            self.average = average
        else:
            raise DeepchecksValueError('average should be one of: none, micro, macro')

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
        self._evals = defaultdict(lambda: {'dice': 0, 'count': 0})

    def update(self, output: Tuple[np.ndarray, np.ndarray]):
        """Update metric with batch of samples."""
        for pred, label in zip(output[0], output[1]):
            label_onehot, pred_onehot = format_segmentation_masks(label, pred, self.threshold)
            if self.average == 'micro':
                tp, label, pred = segmentation_counts_micro(label_onehot, pred_onehot)
                tp_count, label_count, pred_count = np.asarray([tp]), np.asarray([label]), np.asarray([pred])
            else:
                tp_count, label_count, pred_count = segmentation_counts_per_class(label_onehot, pred_onehot)

            dice = (2 * tp_count + self.smooth) / (label_count + pred_count + self.smooth)

            classes_ids = [0] if self.average == 'micro' else np.unique(label)
            for class_id in [int(x) for x in classes_ids]:
                self._evals[class_id]['dice'] += dice[class_id]
                self._evals[class_id]['count'] += 1

    def compute(self):
        """Compute metric value."""
        sorted_classes = [int(class_id) for class_id in sorted(self._evals.keys())]
        max_class = max(sorted_classes)
        scores_per_class = np.empty(max_class + 1) * np.nan

        for class_id in sorted_classes:
            count = self._evals[class_id]['count']
            dice = self._evals[class_id]['dice']
            mean_dice = dice / count if count != 0 else 0
            scores_per_class[class_id] = mean_dice
        if self.average == 'macro':
            scores_per_class = np.nanmean(scores_per_class)
        return scores_per_class


class MeanIoU(Metric):
    """Metric that calculates the mean IoU metric for each class.

    See more: https://en.wikipedia.org/wiki/Jaccard_index

    Parameters
    ----------
    threshold: float, default: 0.5
        prediction value per pixel above which the pixel is considered True.
    smooth: float, default: 1e-3
        smoothing factor to prevent division by zero when the mask is empty.
    average: str, default: none
        The method for averaging over the classes. If none, returns the result per class.
    """

    def __init__(self, *args, threshold: float = 0.5, smooth=1e-3, average: str = 'none', **kwargs):
        super().__init__(*args, **kwargs)
        self._evals = defaultdict(lambda: {'iou': 0, 'count': 0})
        self.threshold = threshold
        self.smooth = smooth
        if average in ['none', 'micro', 'macro']:
            self.average = average
        else:
            raise DeepchecksValueError('average should be one of: none, micro, macro')

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
        self._evals = defaultdict(lambda: {'iou': 0, 'count': 0})

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        """Update metric with batch of samples."""
        for pred, label in zip(output[0], output[1]):
            gt_onehot, pred_onehot = format_segmentation_masks(label, pred, self.threshold)
            tp_count_per_class, gt_count_per_class, pred_count_per_class = segmentation_counts_per_class(
                gt_onehot, pred_onehot)

            iou_per_class = (tp_count_per_class + self.smooth) / \
                            (gt_count_per_class + pred_count_per_class - tp_count_per_class + self.smooth)

            classes_ids = [0] if self.average == 'micro' else np.unique(label)
            for class_id in [int(x) for x in classes_ids]:
                self._evals[class_id]['iou'] += iou_per_class[class_id]
                self._evals[class_id]['count'] += 1

    def compute(self):
        """Compute metric value."""
        sorted_classes = [int(class_id) for class_id in sorted(self._evals.keys())]
        max_class = max(sorted_classes)
        scores_per_class = np.empty(max_class + 1) * np.nan
        for class_id in sorted_classes:
            count = self._evals[class_id]['count']
            iou = self._evals[class_id]['iou']
            mean_iou = iou / count if count != 0 else 0
            scores_per_class[class_id] = mean_iou
        if self.average == 'macro':
            scores_per_class = np.nanmean(scores_per_class)
        return scores_per_class


def per_sample_dice(predictions, labels, threshold: float = 0.5, smooth: float = 1e-3):
    """Calculate Dice score per sample."""
    score = np.empty(len(labels))
    for i in range(len(labels)):
        gt_onehot, pred_onehot = format_segmentation_masks(labels[i], predictions[i], threshold)
        tp_count, gt_count, pred_count = segmentation_counts_micro(gt_onehot, pred_onehot)
        score[i] = (2 * tp_count + smooth) / (gt_count + pred_count + smooth)
    return score.tolist()
