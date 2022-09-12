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
from deepchecks.vision.metrics_utils.semantic_segmentation_metric_utils import format_segmentation_masks, \
    segmentation_counts_per_class


class MeanDice(Metric):
    """Metric that calculates the mean Dice metric for each class.

    See more: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    """

    def __init__(self, threshold: float = 0.5, smooth=1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evals = defaultdict(lambda: {'dice': 0, 'count': 0})
        self.threshold = threshold
        self.smooth = smooth

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
        self._evals = defaultdict(lambda: {'dice': 0, 'count': 0})

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        """Update metric with batch of samples."""
        y_pred, y = output

        for i in range(len(y)):
            gt_onehot, pred_onehot = format_segmentation_masks(y[i], y_pred[i], self.threshold)
            tp_count_per_class, gt_count_per_class, pred_count_per_class = segmentation_counts_per_class(
                gt_onehot, pred_onehot)

            dice_per_class = (2 * tp_count_per_class + self.smooth) / \
                             (gt_count_per_class + pred_count_per_class + self.smooth)

            for class_id in [int(x) for x in torch.unique(y[i])]:
                self._evals[class_id]['dice'] += dice_per_class[class_id]
                self._evals[class_id]['count'] += 1

    def compute(self):
        """Compute metric value."""
        sorted_classes = [int(class_id) for class_id in sorted(self._evals.keys())]
        ret = []
        for class_id in sorted_classes:
            count = self._evals[class_id]['count']
            dice = self._evals[class_id]['dice']
            mean_dice = dice / count if count != 0 else 0
            ret.append(mean_dice)
        return ret



