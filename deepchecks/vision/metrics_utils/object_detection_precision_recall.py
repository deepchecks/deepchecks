from typing import List

import numpy as np

from deepchecks.vision.metrics_utils.detection_precision_recall import AveragePrecision
from deepchecks.vision.metrics_utils.iou_utils import compute_pairwise_ious, jaccard_iou, untorchify


class ObjectDetectionAveragePrecision(AveragePrecision):
    """We are expecting to receive the predictions in the following format: [x, y, w, h, confidence, label]."""

    def get_confidences(self, detection) -> List[float]:
        """Get detections object of single image and should return confidence for each detection."""
        return [d[4].item() for d in detection]

    def calc_pairwise_ious(self, detection, ground_truth) -> np.ndarray:
        """Expect detection and labels of a single image and single class."""
        return compute_pairwise_ious(detection, ground_truth, jaccard_iou)

    def get_detections_classes(self, detection) -> List[int]:
        """Expect torch.Tensor of shape (N, 6) represents detections on single image"""
        return [untorchify(d[5]) for d in detection]

    def get_labels_classes(self, labels) -> List[int]:
        """Get labels of a single image and should return class for each label."""
        return [untorchify(g[0]) for g in labels]

    def get_detection_areas(self, detection) -> List[int]:
        """Get detection object of single image and should return area for each detection."""
        return [d[2].item() * d[3].item() for d in detection]
