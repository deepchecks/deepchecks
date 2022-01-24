from collections import defaultdict

from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import numpy as np
from .iou_utils import compute_ious


class DetectionPrecisionRecall(Metric):
    """"
    We are exepcting to recieve the predictions in the following format:
    [x, y, w, h, confidence, label]
    """
    def __init__(self, iou_threshold=0.5, max_dets=None, area_range=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bb_info = defaultdict(lambda: {"dt": [], "gt": []})
        self.iou_threshold = iou_threshold
        self.max_dets = max_dets
        self.area_range = area_range
        self.i = 0

    @reinit__is_reduced
    def reset(self):
        super(DetectionPrecisionRecall, self).reset()
        self.bb_info = defaultdict(lambda: {"dt": [], "gt": []})
        self.i = 0


    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        for dt, gt in zip(y_pred,y):
            self._group_detections(dt, gt)
            self.i += 1


    @sync_all_reduce("bb_info")
    def compute(self):
        # Calculating pairwise IoUs
        _ious = {k: compute_ious(**v) for k, v in self.bb_info.items()}
        _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})

        for img_id, class_id in self.bb_info:
            ev = self._evaluate_image(
                self.bb_info[img_id, class_id]["dt"],
                self.bb_info[img_id, class_id]["gt"],
                _ious[img_id, class_id]
            )

            acc = _evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])

        # now reduce accumulations
        for class_id in _evals:
            acc = _evals[class_id]
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype(np.bool)
            acc["NP"] = np.sum(acc["NP"])

        res = {}
        # run ap calculation per-class
        for class_id in _evals:
            ev = _evals[class_id]
            res[class_id] = {
                "class": class_id,
                **self._compute_ap_recall(ev["scores"], ev["matched"], ev["NP"])
            }
        return res

    def _group_detections(self, dt, gt):
        """ simply group gts and dts on a imageXclass basis """
        for d in dt:
            c_id = d[5].item()
            self.bb_info[self.i, c_id]["dt"].append(d)
        for g in gt:
            c_id = g[0]
            self.bb_info[self.i, c_id]["gt"].append(g)

    def _evaluate_image(self, det, gt, ious):
        """
        det - [x, y, w, h, confidence, label]
        gt - [label, x, y, w, h]
        """
        # Sort detections by increasing confidence
        det = [self.Prediction(d) for d in det]
        det_sort = np.argsort([-d.confidence for d in det], kind='stable')

        # sort list of dts and chop by max dets
        dt = [det[idx] for idx in det_sort[:self.max_dets]]
        ious = ious[det_sort[:self.max_dets]]

        # generate ignored gt list by area_range
        def _is_ignore(bb):
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

        for d_idx, d in enumerate(dt):
            # information about best match so far (m=-1 -> unmatched)
            iou = min(self.iou_threshold, 1 - 1e-10)
            m = -1
            for g_idx, g in enumerate(gt):
                # if this gt already matched, and not a crowd, continue
                if g_idx in gtm:
                    continue
                # if dt matched to reg gt, and on ignore gt, stop
                if m > -1 and gt_ignore[m] == False and gt_ignore[g_idx] == True:
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

    def _compute_ap_recall(self, scores, matched, NP, recall_thresholds=None):
        """ This curve tracing method has some quirks that do not appear when only unique confidence thresholds
        are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced. """
        if NP == 0:
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

        rc = tp / NP
        pr = tp / (tp + fp)

        # make precision monotonically decreasing
        i_pr = np.maximum.accumulate(pr[::-1])[::-1]

        rec_idx = np.searchsorted(rc, recall_thresholds, side="left")
        n_recalls = len(recall_thresholds)

        # get interpolated precision values at the evaluation thresholds
        i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

        return {
            "precision": pr,
            "recall": rc,
            "AP": np.mean(i_pr),
            "interpolated precision": i_pr,
            "interpolated recall": recall_thresholds,
            "total positives": NP,
            "TP": tp[-1] if len(tp) != 0 else 0,
            "FP": fp[-1] if len(fp) != 0 else 0
        }

    class Prediction:
        def __init__(self, det):
            self.bbox = det[:4]
            self.confidence = det[4]
            self.label = det[5]
