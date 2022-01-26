from collections import defaultdict
import numpy as np


def group_detections(dt, gt):
    """ simply group gts and dts on a imageXclass basis """
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})
    for d in dt:
        c_id = d[5].item()
        bb_info[c_id]["dt"].append(d)
    for g in gt:
        c_id = g[0]
        bb_info[c_id]["gt"].append(g)
    return bb_info


def _jaccard(dt, gt):
    x_dt, y_dt, w_dt, h_dt = dt[:4]
    x_gt, y_gt, w_gt, h_gt = gt[1:]

    x2_dt, y2_dt = x_dt + w_dt, y_dt + h_dt
    x2_gt, y2_gt = x_gt + w_gt, y_gt + h_gt

    # innermost left x
    xi = max(x_dt, x_gt)
    # innermost right x
    x2i = min(x2_dt, x2_gt)
    # same for y
    yi = max(y_dt, y_gt)
    y2i = min(y2_dt, y2_gt)

    # calculate areas
    dt_area = float(w_dt * h_dt)
    gt_area = float(w_gt * h_gt)
    intersection = float(max(x2i - xi, 0)) * float(max(y2i - yi, 0))
    return float(intersection / (dt_area + gt_area - intersection))


def compute_ious(dt, gt):
    """ compute pairwise ious """

    ious = np.zeros((len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            ious[d_idx, g_idx] = _jaccard(d, g)
    return ious