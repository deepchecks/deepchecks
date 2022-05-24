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
from typing import List

import numpy as np

from deepchecks.vision.metrics_utils.detection_precision_recall import AveragePrecisionRecall
from deepchecks.vision.metrics_utils.iou_utils import compute_pairwise_ious, group_class_detection_label, jaccard_iou


class ObjectDetectionAveragePrecision(AveragePrecisionRecall):
    """We are expecting to receive the predictions in the following format: [x, y, w, h, confidence, label]."""

    def get_labels_areas(self, labels) -> List[int]:
        """Get labels object of single image and should return area for each label."""
        return [bbox[3].item() * bbox[4].item() for bbox in labels]

    def group_class_detection_label(self, detections, labels) -> dict:
        """Group detection and labels in dict of format {class_id: {'detected' [...], 'ground_truth': [...] }}."""
        return group_class_detection_label(detections, labels)

    def get_confidences(self, detections) -> List[float]:
        """Get detections object of single image and should return confidence for each detection."""
        return [d[4].item() for d in detections]

    def calc_pairwise_ious(self, detections, labels) -> np.ndarray:
        """Expect detection and labels of a single image and single class."""
        return compute_pairwise_ious(detections, labels, jaccard_iou)

    def get_detection_areas(self, detections) -> List[int]:
        """Get detection object of single image and should return area for each detection."""
        return [d[2].item() * d[3].item() for d in detections]
