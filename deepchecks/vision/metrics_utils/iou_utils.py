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
"""Module for computing Intersection over Unions."""
from collections import defaultdict

import numpy as np
import torch


def jaccard_iou(dt: np.array, gt: np.array):
    """Calculate the jaccard IoU.

    See https://en.wikipedia.org/wiki/Jaccard_index

    Parameters
    ----------
    dt: np.array
        Single Detection in the shape of [x, y, width, height, confidence, class]
    gt: np.array
        Single Ground Truth in the shape of [class, x, y, width, height]
    """
    x_dt, y_dt, w_dt, h_dt = dt[:4]
    x_gt, y_gt, w_gt, h_gt = gt[1:]

    x2_dt, y2_dt = x_dt + w_dt, y_dt + h_dt
    x2_gt, y2_gt = x_gt + w_gt, y_gt + h_gt

    # innermost left x
    xi = x_dt if x_dt > x_gt else x_gt
    # innermost right x
    x2i = x2_dt if x2_dt < x2_gt else x2_gt
    # same for y
    yi = y_dt if y_dt > y_gt else y_gt
    y2i = y2_dt if y2_dt < y2_gt else y2_gt

    # calculate areas
    dt_area = w_dt * h_dt
    gt_area = w_gt * h_gt
    iwidth = x2i - xi if x2i > xi else 0
    ihight = y2i - yi if y2i > yi else 0
    intersection = iwidth * ihight
    return intersection / (dt_area + gt_area - intersection)


def compute_pairwise_ious(detected, ground_truth, iou_func):
    """Compute pairwise ious between detections and ground truth."""
    ious = np.zeros((len(detected), len(ground_truth)))
    for g_idx, g in enumerate(ground_truth):
        for d_idx, d in enumerate(detected):
            ious[d_idx, g_idx] = iou_func(d, g)
    return ious


def group_class_detection_label(detected, ground_truth):
    """Group bounding detection and labels by class."""
    class_bounding_boxes = defaultdict(lambda: {"detected": [], "ground_truth": []})

    for single_detection in detected:
        class_id = untorchify(single_detection[5])
        class_bounding_boxes[class_id]["detected"].append(single_detection.cpu().detach().numpy())
    for single_ground_truth in ground_truth:
        class_id = untorchify(single_ground_truth[0])
        class_bounding_boxes[class_id]["ground_truth"].append(single_ground_truth.cpu().detach().numpy())

    return class_bounding_boxes


def compute_bounding_box_class_ious(detected, ground_truth):
    """Compute ious between bounding boxes of the same class."""
    bb_info = group_class_detection_label(detected, ground_truth)

    # Calculating pairwise IoUs per class
    return {class_id: compute_pairwise_ious(info["detected"], info["ground_truth"], jaccard_iou)
            for class_id, info in bb_info.items()}


def per_sample_mean_iou(predictions, labels):
    """Calculate mean iou for a single sample."""
    mean_ious = []
    for detected, ground_truth in zip(predictions, labels):
        if len(ground_truth) == 0:
            if len(detected) == 0:
                mean_ious.append(1)
            else:
                mean_ious.append(0)
            continue
        elif len(detected) == 0:
            mean_ious.append(0)
            continue

        ious = compute_bounding_box_class_ious(detected, ground_truth)
        count = 0
        sum_iou = 0

        for _, cls_ious in ious.items():
            # Find best fit for each detection
            for detection in cls_ious:
                sum_iou += max(detection, default=0)
                count += 1

        if count:
            mean_ious.append(sum_iou / count)
        else:
            mean_ious.append(0)

    return mean_ious


def untorchify(item):
    """If item is torch tensor do `.item()` else return item itself."""
    if isinstance(item, torch.Tensor):
        return item.cpu().item()
    return item
