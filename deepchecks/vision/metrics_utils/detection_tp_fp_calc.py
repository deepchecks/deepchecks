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
"""Module for calculating detection precision and recall."""
import typing as t
from collections import defaultdict

import numpy as np
import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from deepchecks.vision.metrics_utils.metric_mixin import MetricMixin, ObjectDetectionMetricMixin


class TpFp(Metric, MetricMixin):
    """Abstract class to calculate average precision and recall for various vision tasks.

    Parameters
    ----------
    max_dets: Union[List[int], Tuple[int]], default: [1, 10, 100]
        Maximum number of detections per class.
    area_range: tuple, default: (32**2, 96**2)
        Slices for small/medium/large buckets.
    return_option: int, default: 0
        0: ap only, 1: ar only, None: all (not ignite complient)
    """

    def __init__(self, *args, iou_thres: float = 0.5, confidence_thres: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)

        self._evals = defaultdict(lambda: {"matched": [], "NP": 0})

        self.iou_thres = iou_thres
        self.confidence_thres = confidence_thres
        self._i = 0

    @reinit__is_reduced
    def reset(self):
        """Reset metric state."""
        super().reset()
        self._evals = defaultdict(lambda: {"matched": [], "NP": 0})
        self._i = 0

    @reinit__is_reduced
    def update(self, output):
        """Update metric with batch of samples."""
        y_pred, y = output

        for detected, ground_truth in zip(y_pred, y):
            if isinstance(detected, torch.Tensor):
                detected = detected.cpu()
            if isinstance(ground_truth, torch.Tensor):
                ground_truth = ground_truth.cpu()

            self._group_detections(detected, ground_truth)
            self._i += 1

    @sync_all_reduce("_evals")
    def compute(self):
        """Compute metric value."""
        # now reduce accumulations
        sorted_classes = [int(class_id) for class_id in sorted(self._evals.keys())]
        max_class = max(sorted_classes)
        res = -np.ones(max_class + 1)
        for class_id in sorted_classes:
            ev = self._evals[class_id]
            res[class_id] = self._compute_fp_tp_fn(np.array(ev["matched"]), ev["NP"])
        return res

    def _group_detections(self, detected, ground_truth):
        """Group gts and dts on a imageXclass basis."""
        # Calculating pairwise IoUs on classes
        bb_info = self.group_class_detection_label(detected, ground_truth)
        ious = {k: self.calc_pairwise_ious(v["detected"], v["ground_truth"]) for k, v in bb_info.items()}

        for class_id in ious.keys():
            matched, n_positives = self._evaluate_image(
                self.get_confidences(bb_info[class_id]["detected"]),
                bb_info[class_id]["ground_truth"],
                ious[class_id]
            )

            acc = self._evals[class_id]
            acc["matched"] += matched
            acc["NP"] += n_positives

    def _evaluate_image(self, confidences: t.List[float], ground_truths: t.List, ious: np.ndarray) -> \
            t.Tuple[t.List[float], t.List[bool], int]:
        """Evaluate image."""
        # Sort detections by decreasing confidence
        sorted_confidence_ids = np.argsort(confidences, kind="stable")[::-1] > self.confidence_thres
        orig_ious = ious

        # sort list of dts and chop by max dets
        ious = orig_ious[sorted_confidence_ids]

        detection_matches = self._get_best_matches(ground_truths, ious)
        matched = [d_idx in detection_matches for d_idx in range(len(ious))]

        return matched, len(ground_truths)

    def _get_best_matches(self, ground_truths: t.List, ious: np.ndarray) -> t.Dict[int, int]:
        ground_truth_matched = {}
        detection_matches = {}

        for d_idx in range(len(ious)):
            # information about best match so far (best_match=-1 -> unmatched)
            best_iou = min(self.iou_thres, 1 - 1e-10)
            best_match = -1
            for g_idx in range(len(ground_truths)):
                # if this gt already matched, continue
                if g_idx in ground_truth_matched:
                    continue

                if ious[d_idx, g_idx] >= best_iou:
                    best_iou = ious[d_idx, g_idx]
                    best_match = g_idx
            if best_match != -1:
                detection_matches[d_idx] = best_match
                ground_truth_matched[best_match] = d_idx
        return detection_matches

    def _compute_fp_tp(self, matched: t.List[bool], n_positives: int):
        tp = np.sum(matched)
        fp = np.sum(~matched)

        return tp, fp, n_positives


class ObjectDetectionTpFp(TpFp, ObjectDetectionMetricMixin):
    """We are expecting to receive the predictions in the following format: [x, y, w, h, confidence, label]."""
