# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module for average precision for object detection."""
from collections import defaultdict
from typing import List

import numpy as np

from deepchecks.vision.metrics_utils.detection_precision_recall import AveragePrecision
from deepchecks.vision.metrics_utils.iou_utils import compute_pairwise_ious, jaccard_iou, untorchify


class ObjectDetectionAveragePrecision(AveragePrecision):
    """We are expecting to receive the predictions in the following format: [x, y, w, h, confidence, label]."""

    def group_class_detection_label(self, detection, ground_truth) -> dict:
        """Group detection and labels in dict of format {class_id: {'detected' [...], 'ground_truth': [...] }}."""
        class_bounding_boxes = defaultdict(lambda: {"detected": [], "ground_truth": []})

        for single_detection in detection:
            class_id = untorchify(single_detection[5])
            class_bounding_boxes[class_id]["detected"].append(single_detection)
        for single_ground_truth in ground_truth:
            class_id = untorchify(single_ground_truth[0])
            class_bounding_boxes[class_id]["ground_truth"].append(single_ground_truth)

        return class_bounding_boxes

    def get_confidences(self, detection) -> List[float]:
        """Get detections object of single image and should return confidence for each detection."""
        return [d[4].item() for d in detection]

    def calc_pairwise_ious(self, detection, ground_truth) -> np.ndarray:
        """Expect detection and labels of a single image and single class."""
        return compute_pairwise_ious(detection, ground_truth, jaccard_iou)

    def get_detection_areas(self, detection) -> List[int]:
        """Get detection object of single image and should return area for each detection."""
        return [d[2].item() * d[3].item() for d in detection]
