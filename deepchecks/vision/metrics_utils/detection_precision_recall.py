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
from collections import defaultdict
from typing import List, Tuple

from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from sklearn.metrics import mean_absolute_error
import torch
import numpy as np
from .iou_utils import compute_ious


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


class AveragePrecision(Metric):
    """We are expecting to receive the predictions in the following format: [x, y, w, h, confidence, label].

    Parameters
    ----------
    num_classes : int
        Number of classes.
    iou_threshold : float , default: 0.5
        Intersection over area threshold.
    max_dets: int, default: None
        Maximum number of detections per class.
    area_range: tuple, default: (32**2, 96**2)
        Slices for small/medium/large buckets.
    return_ap_only: bool, default: True
        If True, only the average precision will be returned.
    """

    def __init__(self, *args, max_dets: List[int] = (1, 10, 100),
                 area_range: Tuple = (32**2, 96**2), return_single_value: bool = True,
                 return_ap_only: bool = True,
                 only_per_class: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        self.only_per_class = only_per_class
        self.return_ap_only = return_ap_only
        self.return_single_value = return_single_value
        if return_single_value:
            max_dets = [max_dets[-1]]
            print(max_dets)
            self.area_ranges_names = ["all"]
        else:
            self.area_ranges_names = ["small", "medium", "large", "all"]
        print('ahhh')
        self.iou_thresholds = np.arange(0.5, 0.95, 0.05)
        self.max_dets = max_dets
        self.area_range = area_range
        self.i = 0

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

        for dt, gt in zip(y_pred, y):
            self._group_detections(dt, gt)
            self.i += 1

    @sync_all_reduce("_evals")
    def compute(self):
        """Compute metric value."""
        # now reduce accumulations
        for class_id in self._evals:
            acc = self._evals[class_id]
            acc["scores"] = _dict_conc(acc["scores"])
            acc["matched"] = _dict_conc(acc["matched"])
            acc["NP"] = _dict_conc(acc["NP"])
        reses = {}
        for min_iou in self.iou_thresholds:
            for dets in self.max_dets:
                for area_size in self.area_ranges_names:
                    res = {}
                    if not reses.get((area_size, dets, "0.5..0.95")):
                        reses[(area_size, dets, "0.5..0.95")] = {}
                    # run ap calculation per-class
                    for class_id in self._evals:
                        ev = self._evals[class_id]
                        res[class_id] = {
                            "class": class_id,
                            **self._compute_ap_recall(np.array(ev["scores"][(area_size, dets, min_iou)]),
                                                      np.array(ev["matched"][(area_size, dets, min_iou)]),
                                                      np.sum(np.array(ev["NP"][(area_size, dets, min_iou)])))
                        }
                        if not reses[(area_size, dets, "0.5..0.95")].get(class_id):
                            reses[(area_size, dets, "0.5..0.95")][class_id] = {
                                "class": class_id,
                                "precision": [res[class_id]["precision"]],
                                "recall": [res[class_id]["recall"]]
                            }
                        else:
                            reses[(area_size, dets, "0.5..0.95")][class_id]["precision"].append(res[class_id]["precision"])
                            reses[(area_size, dets, "0.5..0.95")][class_id]["recall"].append(res[class_id]["recall"])
                    if not self.only_per_class:
                        all_evals = _dict_conc(self._evals)
                        res[-1] = {
                                "class": -1,
                                **self._compute_ap_recall(np.array(all_evals[(area_size, dets, min_iou)]),
                                                          np.array(all_evals[(area_size, dets, min_iou)]),
                                                          np.sum(np.array(all_evals[(area_size, dets, min_iou)])))
                        }
                        if not reses[(area_size, dets, "0.5..0.95")].get(-1):
                            reses[(area_size, dets, "0.5..0.95")][-1] = {
                                "class": -1,
                                "precision": [res[-1]["precision"]],
                                "recall": [res[-1]["recall"]]
                            }
                        else:
                            reses[(area_size, dets, "0.5..0.95")][-1]["precision"].append(res[-1]["precision"])
                            reses[(area_size, dets, "0.5..0.95")][-1]["recall"].append(res[-1]["recall"])
                    if self.return_ap_only:
                        res = torch.tensor([res[k]["precision"] for k in sorted(res.keys())])
                    if dets == 100 and area_size == 'all':
                        reses[(area_size, dets, min_iou)] = res
        for dets in self.max_dets:
            for area_size in self.area_ranges_names:
                for class_id in self._evals:
                    reses[(area_size, dets, "0.5..0.95")][class_id]["precision"] = \
                        np.mean(reses[(area_size, dets, "0.5..0.95")][class_id]["precision"])
                    reses[(area_size, dets, "0.5..0.95")][class_id]["recall"] = \
                            np.mean(reses[(area_size, dets, "0.5..0.95")][class_id]["recall"])
                if self.return_ap_only:
                    reses[(area_size, dets, "0.5..0.95")] = \
                        torch.tensor([reses[(area_size, dets, "0.5..0.95")][k]["precision"]
                                        for k in sorted(reses[(area_size, dets, "0.5..0.95")].keys())])
        if self.return_single_value:
            return reses[(self.area_ranges_names[0], self.max_dets[0], "0.5..0.95")]
        return [reses]

    def _group_detections(self, dt, gt):
        """Group gts and dts on a imageXclass basis."""
        bb_info = defaultdict(lambda: {"dt": [], "gt": []})

        for d in dt:
            c_id = d[5].item()
            bb_info[c_id]["dt"].append(d)
        for g in gt:
            c_id = g[0]
            bb_info[c_id]["gt"].append(g)

        # Calculating pairwise IoUs
        ious = {k: compute_ious(**v) for k, v in bb_info.items()}

        for class_id in ious.keys():
            ev = self._evaluate_image(
                bb_info[class_id]["dt"],
                bb_info[class_id]["gt"],
                ious[class_id]
            )

            acc = self._evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])

    def _evaluate_image(self, det, gt, ious):
        """Det - [x, y, w, h, confidence, label], gt - [label, x, y, w, h]."""
        # Sort detections by increasing confidence
        det = [self.Prediction(d) for d in det]
        det_sort = np.argsort([-d.confidence for d in det], kind="stable")
        orig_ious = ious
        orig_gt = gt

        scores = {}
        matched = {}
        n_gts = {}
        for min_iou in self.iou_thresholds:
            for dets in self.max_dets:
                for area_size in ["small", "medium", "large", "all"]:
                    # sort list of dts and chop by max dets
                    dt = [det[idx] for idx in det_sort[:dets]]
                    ious = orig_ious[det_sort[:dets]]
                    gt_ignore = [self._is_ignore_area(g[3] * g[4], area_size) for g in orig_gt]

                    # sort gts by ignore last
                    gt_sort = np.argsort(gt_ignore, kind="stable")
                    gt = [orig_gt[idx] for idx in gt_sort]
                    gt_ignore = [gt_ignore[idx] for idx in gt_sort]

                    ious = ious[:, gt_sort]

                    gtm = {}
                    dtm = {}

                    for d_idx, _ in enumerate(dt):
                        # information about best match so far (m=-1 -> unmatched)
                        iou = min(min_iou, 1 - 1e-10)
                        m = -1
                        for g_idx, _ in enumerate(gt):
                            # if this gt already matched, and not a crowd, continue
                            if g_idx in gtm:
                                continue
                            # if dt matched to reg gt, and on ignore gt, stop
                            if m > -1 and not gt_ignore[m] and gt_ignore[g_idx]:
                                break
                            # continue to next gt unless better match made
                            if ious[d_idx, g_idx] < iou:
                                continue
                            # if match successful and best so far, store appropriately
                            iou = ious[d_idx, g_idx]
                            m = g_idx
                        # if match made store id of match for both dt and gt
                        if m == -1:
                            continue
                        dtm[d_idx] = m
                        gtm[m] = d_idx

                    # generate ignore list for dts
                    dt_ignore = [
                        gt_ignore[dtm[d_idx]] if d_idx in dtm
                        else self._is_ignore_area(d.bbox[2] * d.bbox[3], area_size)
                        for d_idx, d in enumerate(dt)
                    ]

                    # get score for non-ignored dts
                    scores[(area_size, dets, min_iou)] = [dt[d_idx].confidence for d_idx in range(len(dt))
                                                          if not dt_ignore[d_idx]]
                    matched[(area_size, dets, min_iou)] = [d_idx in dtm for d_idx in range(len(dt))
                                                           if not dt_ignore[d_idx]]

                    n_gts[(area_size, dets, min_iou)] = len([g_idx for g_idx in range(len(gt)) if not gt_ignore[g_idx]])
        return {"scores": scores, "matched": matched, "NP": n_gts}

    def _compute_ap_recall(self, scores, matched, n_positives, recall_thresholds=None):
        if n_positives == 0:
            return {
                "precision": 0,
                "recall": 0
            }

        # by default evaluate on 101 recall levels
        if recall_thresholds is None:
            recall_thresholds = np.linspace(0.0,
                                            1.00,
                                            int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                            endpoint=True)

        # sort in descending score order
        inds = np.argsort(-scores, kind="stable")

        scores = scores[inds]
        matched = matched[inds]

        if len(matched):
            tp = np.cumsum(matched)
            fp = np.cumsum(~matched)

            rc = tp / n_positives
            pr = tp / (tp + fp)

            # make precision monotonically decreasing
            i_pr = np.maximum.accumulate(pr[::-1])[::-1]

            rec_idx = np.searchsorted(rc, recall_thresholds, side="left")

            # get interpolated precision values at the evaluation thresholds
            i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

            return {
                "precision": np.mean(i_pr),
                "recall": tp[-1] / n_positives
            }

        return {
            "precision": 0,
            "recall": 0
        }

    def _is_ignore_area(self, area_bb, area_size):
        """Generate ignored gt list by area_range."""
        if self.return_single_value:
            return False
        if area_size == "small":
            return not area_bb < self.area_range[0]
        if area_size == "medium":
            return not self.area_range[0] <= area_bb <= self.area_range[1]
        if area_size == "large":
            return not area_bb > self.area_range[1]
        return False

    class Prediction:
        """A class defining the prediction of a single image in an object detection task."""

        def __init__(self, det):
            self.bbox = det[:4]
            self.confidence = det[4]
            self.label = det[5]
