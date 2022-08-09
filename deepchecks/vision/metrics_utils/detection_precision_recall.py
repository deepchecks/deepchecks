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
import warnings
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from deepchecks.vision.metrics_utils.metric_mixin import MetricMixin, ObjectDetectionMetricMixin


def _dict_conc(test_list):
    result = defaultdict(list)

    for i in range(len(test_list)):
        current = test_list[i]
        for key, value in current.items():
            if isinstance(value, list):
                for j in range(len(value)):
                    result[key].append(value[j])
            else:
                result[key].append(value)

    return result


class AveragePrecisionRecall(Metric, MetricMixin):
    """Abstract class to calculate average precision and recall for various vision tasks.

    Parameters
    ----------
    max_dets: Union[List[int], Tuple[int]], default: [1, 10, 100]
        Maximum number of detections per class.
    area_range: tuple, default: (32**2, 96**2)
        Slices for small/medium/large buckets.
    return_option: str, default: 'ap'
        ap: ap only, ar: ar only, None: all (not ignite compliant)
    """

    def __init__(self, *args, max_dets: Union[List[int], Tuple[int]] = (1, 10, 100),
                 area_range: Tuple = (32**2, 96**2),
                 return_option: str = "ap", **kwargs):
        super().__init__(*args, **kwargs)

        self.return_option = return_option
        if self.return_option is not None:
            max_dets = [max_dets[-1]]
            self.area_ranges_names = ["all"]
        else:
            self.area_ranges_names = ["small", "medium", "large", "all"]
        self.iou_thresholds = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.max_detections_per_class = max_dets
        self.area_range = area_range

    @reinit__is_reduced
    def reset(self):
        """Reset metric state."""
        super().reset()
        self._evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        self.i = 0

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
            self.i += 1

    @sync_all_reduce("_evals")
    def compute(self):
        """Compute metric value."""
        # now reduce accumulations
        sorted_classes = [int(class_id) for class_id in sorted(self._evals.keys())]
        max_class = max(sorted_classes)
        for class_id in sorted_classes:
            acc = self._evals[class_id]
            acc["scores"] = _dict_conc(acc["scores"])
            acc["matched"] = _dict_conc(acc["matched"])
            acc["NP"] = _dict_conc(acc["NP"])
        reses = {"precision": -np.ones((len(self.iou_thresholds),
                                        len(self.area_ranges_names),
                                        len(self.max_detections_per_class),
                                        max_class + 1)),
                 "recall": -np.ones((len(self.iou_thresholds),
                                     len(self.area_ranges_names),
                                     len(self.max_detections_per_class),
                                     max_class + 1))}
        for iou_i, min_iou in enumerate(self.iou_thresholds):
            for dets_i, dets in enumerate(self.max_detections_per_class):
                for area_i, area_size in enumerate(self.area_ranges_names):
                    precision_list = np.empty(max_class + 1)
                    precision_list.fill(np.nan)
                    recall_list = np.empty(max_class + 1)
                    recall_list.fill(np.nan)
                    # run ap calculation per-class
                    for class_id in sorted_classes:
                        ev = self._evals[class_id]
                        precision, recall = \
                            self._compute_ap_recall(np.array(ev["scores"][(area_size, dets, min_iou)]),
                                                    np.array(ev["matched"][(area_size, dets, min_iou)]),
                                                    np.sum(np.array(ev["NP"][(area_size, dets, min_iou)])))
                        precision_list[class_id] = precision
                        recall_list[class_id] = recall
                    reses["precision"][iou_i, area_i, dets_i] = precision_list
                    reses["recall"][iou_i, area_i, dets_i] = recall_list
        if self.return_option == "ap":
            return torch.tensor(self.get_classes_scores_at(reses["precision"],
                                                           max_dets=self.max_detections_per_class[0],
                                                           area=self.area_ranges_names[0],
                                                           get_mean_val=False))
        elif self.return_option == "ar":
            return torch.tensor(self.get_classes_scores_at(reses["recall"],
                                                           max_dets=self.max_detections_per_class[0],
                                                           area=self.area_ranges_names[0],
                                                           get_mean_val=False))
        return [reses]

    def _group_detections(self, detected, ground_truth):
        """Group gts and dts on a imageXclass basis."""
        # Calculating pairwise IoUs on classes
        bb_info = self.group_class_detection_label(detected, ground_truth)
        ious = {k: self.calc_pairwise_ious(v["detected"], v["ground_truth"]) for k, v in bb_info.items()}

        for class_id in ious.keys():
            image_evals = self._evaluate_image(
                bb_info[class_id]["detected"],
                bb_info[class_id]["ground_truth"],
                ious[class_id]
            )

            acc = self._evals[class_id]
            acc["scores"].append(image_evals["scores"])
            acc["matched"].append(image_evals["matched"])
            acc["NP"].append(image_evals["NP"])

    def _evaluate_image(self, detections, ground_truths, ious):
        """Evaluate image."""
        # Sort detections by decreasing confidence
        confidences = self.get_confidences(detections)
        areas = self.get_detection_areas(detections)
        sorted_confidence_ids = np.argsort(confidences, kind="stable")[::-1]
        orig_ious = ious
        orig_gt = ground_truths
        ground_truth_area = np.array(self.get_labels_areas(ground_truths))

        scores = {}
        matched = {}
        n_gts = {}
        for min_iou in self.iou_thresholds:
            for top_n_detections in self.max_detections_per_class:
                for area_size in self.area_ranges_names:
                    # sort list of dts and chop by max dets
                    top_detections_idx = sorted_confidence_ids[:top_n_detections]
                    ious = orig_ious[top_detections_idx]
                    ground_truth_to_ignore = [self._is_ignore_area(gt_area, area_size) for gt_area in ground_truth_area]

                    # sort gts by ignore last
                    gt_sort = np.argsort(ground_truth_to_ignore, kind="stable")
                    ground_truths = [orig_gt[idx] for idx in gt_sort]
                    ground_truth_to_ignore = [ground_truth_to_ignore[idx] for idx in gt_sort]

                    ious = ious[:, gt_sort]

                    detection_matches = \
                        self._get_best_matches(top_detections_idx, min_iou, ground_truths, ground_truth_to_ignore, ious)

                    # generate ignore list for dts
                    detections_to_ignore = [
                        ground_truth_to_ignore[detection_matches[d_idx]] if d_idx in detection_matches
                        else self._is_ignore_area(areas[real_index], area_size)
                        for d_idx, real_index in enumerate(top_detections_idx)
                    ]

                    # get score for non-ignored dts
                    scores[(area_size, top_n_detections, min_iou)] = \
                        [confidences[real_index] for d_idx, real_index in enumerate(top_detections_idx)
                         if not detections_to_ignore[d_idx]]
                    matched[(area_size, top_n_detections, min_iou)] = \
                        [d_idx in detection_matches for d_idx, real_index in enumerate(top_detections_idx)
                         if not detections_to_ignore[d_idx]]
                    n_gts[(area_size, top_n_detections, min_iou)] = \
                        len([g_idx for g_idx in range(len(ground_truths)) if not ground_truth_to_ignore[g_idx]])
        return {"scores": scores, "matched": matched, "NP": n_gts}

    def _get_best_matches(self, dt, min_iou, ground_truths, ground_truth_to_ignore, ious):
        ground_truth_matched = {}
        detection_matches = {}

        for d_idx in range(len(dt)):
            # information about best match so far (best_match=-1 -> unmatched)
            best_iou = min(min_iou, 1 - 1e-10)
            best_match = -1
            for g_idx in range(len(ground_truths)):
                # if this gt already matched, continue
                if g_idx in ground_truth_matched:
                    continue
                # if dt matched and currently on ignore gt, stop
                # this exists to allow for matching ignored ground truth, so that we ignore this detection
                if best_match > -1 and ground_truth_to_ignore[g_idx]:
                    break

                if ious[d_idx, g_idx] >= best_iou:
                    best_iou = ious[d_idx, g_idx]
                    best_match = g_idx
            if best_match != -1:
                detection_matches[d_idx] = best_match
                ground_truth_matched[best_match] = d_idx
        return detection_matches

    def _compute_ap_recall(self, scores, matched, n_positives, recall_thresholds=None):
        if n_positives == 0:
            return -1, -1

        # by default evaluate on 101 recall levels
        if recall_thresholds is None:
            recall_thresholds = np.linspace(0.0,
                                            1.00,
                                            int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                            endpoint=True)

        # sort in descending score order
        inds = np.argsort(-scores, kind="mergesort")

        scores = scores[inds]
        matched = matched[inds]

        if len(matched):
            tp = np.cumsum(matched)
            fp = np.cumsum(~matched)
            rc = tp / n_positives
            pr = tp / (tp + fp + np.spacing(1))

            # make precision monotonically decreasing
            i_pr = np.maximum.accumulate(pr[::-1])[::-1]

            rec_idx = np.searchsorted(rc, recall_thresholds, side="left")

            # get interpolated precision values at the evaluation thresholds
            i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

            return np.mean(i_pr), rc[-1]
        return 0, 0

    def _is_ignore_area(self, area_bb, area_size):
        """Generate ignored gt list by area_range."""
        if area_size == "small":
            return not area_bb < self.area_range[0]
        if area_size == "medium":
            return not self.area_range[0] <= area_bb <= self.area_range[1]
        if area_size == "large":
            return not area_bb > self.area_range[1]
        return False

    def filter_res(self, res: np.ndarray, iou: float = None, area: str = None, max_dets: int = None):
        """Get the value of a result by the filtering values.

        Parameters
        ----------
        res: np.array
            either precision or recall when using the '2' return option
        iou : float, default: None
            filter by iou threshold
        area : str, default: None
            filter by are range name ["small", "medium", "large", "all"]
        max_dets : int, default: None
            filter by max detections

        Returns
        -------
        np.array
           The filtered result.
        """
        if iou:
            iou_i = [i for i, iou_thres in enumerate(self.iou_thresholds) if iou == iou_thres]
            res = res[iou_i, :, :, :]
        if area:
            area_i = [i for i, area_name in enumerate(self.area_ranges_names) if area == area_name]
            res = res[:, area_i, :, :]
        if max_dets:
            dets_i = [i for i, det in enumerate(self.max_detections_per_class) if max_dets == det]
            res = res[:, :, dets_i, :]
        return res

    def get_classes_scores_at(self, res: np.ndarray, iou: float = None, area: str = None, max_dets: int = None,
                              get_mean_val: bool = True, zeroed_negative: bool = True):
        """Get the mean value of the classes scores and the result values.

        Parameters
        ----------
        res: np.array
            either precision or recall when using the '2' return option
        iou : float, default: None
            filter by iou threshold
        area : str, default: None
            filter by are range name ["small", "medium", "large", "all"]
        max_dets : int, default: None
            filter by max detections
        get_mean_val : bool, default: True
            get mean value if True, if False get per class
        zeroed_negative : bool, default: True
            if getting the class results list set negative (-1) values to 0

        Returns
        -------
        Union[List[float], float]
           The mean value of the classes scores or the scores list.
        """
        res = self.filter_res(res, iou, area, max_dets)
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            res = np.nanmean(res[:, :, :], axis=0)
            if get_mean_val:
                return np.nanmean(res[res > -1])
            if zeroed_negative:
                res = res.clip(min=0)
            return res[0][0]


class ObjectDetectionAveragePrecision(AveragePrecisionRecall, ObjectDetectionMetricMixin):
    """Calculate average precision and recall for object detection.

    Parameters
    ----------
    max_dets: Union[List[int], Tuple[int]], default: [1, 10, 100]
        Maximum number of detections per class.
    area_range: tuple, default: (32**2, 96**2)
        Slices for small/medium/large buckets.
    return_option: str, default: 'ap'
        ap: ap only, ar: ar only, None: all (not ignite complient)
    """
