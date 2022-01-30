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
"""Module for calculating detection precision and recall."""
from collections import defaultdict

from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import numpy as np
from .iou_utils import compute_ious


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
    area_range: tuple, default: None
        Range of image area to be evaluated.
    return_ap_only: bool, default: True
        If True, only the average precision will be returned.
    """

    def __init__(self, *args, iou_threshold=0.5, max_dets=None, area_range=None, return_ap_only: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        self.iou_threshold = iou_threshold
        self.max_dets = max_dets
        self.area_range = area_range
        self.return_ap_only = return_ap_only
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
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype(np.bool)
            acc["NP"] = np.sum(acc["NP"])

        res = {}
        # run ap calculation per-class
        for class_id in self._evals:
            ev = self._evals[class_id]
            res[class_id] = {
                "class": class_id,
                **self._compute_ap_recall(ev["scores"], ev["matched"], ev["NP"])
            }
        if self.return_ap_only:
            res = torch.tensor([res[k]["AP"] for k in sorted(res.keys())])
        return res

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

        # sort list of dts and chop by max dets
        dt = [det[idx] for idx in det_sort[:self.max_dets]]
        ious = ious[det_sort[:self.max_dets]]

        # generate ignored gt list by area_range
        def _is_ignore(bb):  # pylint: disable=unused-argument
            return False
            # TODO: Calculate the area of the bbox and filter
            # if self.area_range is None:
            #     return False
            # return not (self.area_range[0] <= _get_area(bb) <= area_range[1])

        gt_ignore = [_is_ignore(g) for g in gt]

        # sort gts by ignore last
        gt_sort = np.argsort(gt_ignore, kind="stable")
        gt = [gt[idx] for idx in gt_sort]
        gt_ignore = [gt_ignore[idx] for idx in gt_sort]

        ious = ious[:, gt_sort]

        gtm = {}
        dtm = {}

        for d_idx, _ in enumerate(dt):
            # information about best match so far (m=-1 -> unmatched)
            iou = min(self.iou_threshold, 1 - 1e-10)
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
            gt_ignore[dtm[d_idx]] if d_idx in dtm else _is_ignore(d) for d_idx, d in enumerate(dt)
        ]

        # get score for non-ignored dts
        scores = [dt[d_idx].confidence for d_idx in range(len(dt)) if not dt_ignore[d_idx]]
        matched = [d_idx in dtm for d_idx in range(len(dt)) if not dt_ignore[d_idx]]

        n_gts = len([g_idx for g_idx in range(len(gt)) if not gt_ignore[g_idx]])
        return {"scores": scores, "matched": matched, "NP": n_gts}

    def _compute_ap_recall(self, scores, matched, n_positives, recall_thresholds=None):
        if n_positives == 0:
            return {
                "precision": None,
                "recall": None,
                "AP": None,
                "interpolated precision": None,
                "interpolated recall": None,
                "total positives": None,
                "TP": None,
                "FP": None
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
            "precision": pr,
            "recall": rc,
            "AP": np.mean(i_pr),
            "interpolated precision": i_pr,
            "interpolated recall": recall_thresholds,
            "total positives": n_positives,
            "TP": tp[-1] if len(tp) != 0 else 0,
            "FP": fp[-1] if len(fp) != 0 else 0
        }

    class Prediction:
        """A class defining the prediction of a single image in an object detection task."""

        def __init__(self, det):
            self.bbox = det[:4]
            self.confidence = det[4]
            self.label = det[5]
