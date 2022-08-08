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
"""Module for calculating verious detection metrics."""
import typing as t
from collections import defaultdict

import numpy as np
import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from deepchecks.utils.metrics import averaging_mechanism
from deepchecks.vision.metrics_utils.confusion_matrix_counts_metrics import AVAILABLE_EVALUTING_FUNCTIONS
from deepchecks.vision.metrics_utils.metric_mixin import MetricMixin, ObjectDetectionMetricMixin


class TpFpFn(Metric, MetricMixin):
    """Abstract class to calculate the TP, FP, FN and runs an evaluating function on the result.

    Parameters
    ----------
    iou_thres: float, default: 0.5
        IoU below this threshold will be ignored.
    confidence_thres: float, default: 0.5
        Confidence below this threshold will be ignored.
    evaluating_function: Union[Callable, str], default: "recall"
        will run on each class result i.e `func(tp, fp, fn)`
    averaging_method : str, default: 'per_class'
        Determines which averaging method to apply, possible values are:
        'per_class': Return a np array with the scores for each class (sorted by class name).
        'binary': Returns the score for the positive class. Should be used only in binary classification cases.
        'micro': Returns the micro-averaged score.
        'macro': Returns the mean of scores per class.
        'weighted': Returns a weighted mean of scores based of the class size in y_true.
    """

    def __init__(self, *args, iou_thres: float = 0.5, confidence_thres: float = 0.5,
                 evaluating_function: t.Union[t.Callable, str] = "recall", averaging_method="per_class", **kwargs):
        super().__init__(*args, **kwargs)

        self.iou_thres = iou_thres
        self.confidence_thres = confidence_thres
        if isinstance(evaluating_function, str):
            evaluating_function = AVAILABLE_EVALUTING_FUNCTIONS.get(evaluating_function)
            if evaluating_function is None:
                raise ValueError(
                    f"Expected evaluating_function one of {list(AVAILABLE_EVALUTING_FUNCTIONS.keys())},"
                    f" received: {evaluating_function}")
        self.evaluating_function = evaluating_function
        self.averaging_method = averaging_method

    @reinit__is_reduced
    def reset(self):
        """Reset metric state."""
        super().reset()
        self._evals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        self._i = 0

    @reinit__is_reduced
    def update(self, output):
        """Update metric with batch of samples."""
        y_pred, y = output

        for detected, ground_truth in zip(y_pred, y):
            if isinstance(detected, torch.Tensor):
                detected = detected.cpu().detach()
            if isinstance(ground_truth, torch.Tensor):
                ground_truth = ground_truth.cpu().detach()

            self._group_detections(detected, ground_truth)
            self._i += 1

    @sync_all_reduce("_evals")
    def compute(self):
        """Compute metric value."""
        # now reduce accumulations
        sorted_classes = [int(class_id) for class_id in sorted(self._evals.keys())]
        max_class = max(sorted_classes)

        if self.averaging_method == "micro":
            tp, fp, fn = 0, 0, 0
            # Classes that did not appear in the data are not considered as part of micro averaging.
            for class_id in sorted_classes:
                ev = self._evals[class_id]
                tp, fp, fn = tp + ev["tp"], fp + ev["fp"], fn + ev["fn"]
            return self.evaluating_function(tp, fp, fn)

        scores_per_class, weights = -np.ones(max_class + 1), np.zeros(max_class + 1)
        for class_id in sorted_classes:
            ev = self._evals[class_id]
            scores_per_class[class_id] = self.evaluating_function(ev["tp"], ev["fp"], ev["fn"])
            weights[class_id] = ev["tp"] + ev["fn"]

        return averaging_mechanism(self.averaging_method, scores_per_class, weights)

    def _group_detections(self, detected, ground_truth):
        """Group gts and dts on a imageXclass basis."""
        # Calculating pairwise IoUs on classes
        bb_info = self.group_class_detection_label(detected, ground_truth)
        ious = {k: self.calc_pairwise_ious(v["detected"], v["ground_truth"]) for k, v in bb_info.items()}

        for class_id in ious.keys():
            tp, fp, fn = self._evaluate_image(
                np.array(self.get_confidences(bb_info[class_id]["detected"])),
                bb_info[class_id]["ground_truth"],
                ious[class_id]
            )

            acc = self._evals[class_id]
            acc["tp"] += tp
            acc["fp"] += fp
            acc["fn"] += fn

    def _evaluate_image(self, confidences: t.List[float], ground_truths: t.List, ious: np.ndarray) -> \
            t.Tuple[float, float, float]:
        """Evaluate image."""
        # Sort detections by decreasing confidence
        confidences = confidences[confidences > self.confidence_thres]
        sorted_confidence_ids = np.argsort(confidences, kind="stable")[::-1]
        orig_ious = ious

        # sort list of dts and chop by max dets
        ious = orig_ious[sorted_confidence_ids]

        detection_matches = self._get_best_matches(ground_truths, ious)
        matched = np.array([d_idx in detection_matches for d_idx in range(len(ious))])
        if len(matched) == 0:
            tp, fp = 0, 0
        else:
            tp = np.sum(matched)
            fp = len(matched) - tp
        return tp, fp, len(ground_truths) - tp

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


class ObjectDetectionTpFpFn(TpFpFn, ObjectDetectionMetricMixin):
    """Calculate the TP, FP, FN and runs an evaluating function on the result.

    Parameters
    ----------
    iou_thres: float, default: 0.5
        Threshold of the IoU.
    confidence_thres: float, default: 0.5
        Threshold of the confidence.
    evaluating_function: Union[Callable, str], default: "recall"
        will run on each class result i.e `func(tp, fp, fn)`
    """
