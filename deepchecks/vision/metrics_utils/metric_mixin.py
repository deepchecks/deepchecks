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
"""Module for Metric Mixin."""
import typing as t
from abc import abstractmethod

import numpy as np

from deepchecks.vision.metrics_utils.iou_utils import compute_pairwise_ious, group_class_detection_label, jaccard_iou


class MetricMixin:
    @abstractmethod
    def get_confidences(self, detections) -> t.List[float]:
        """Get detections object of single image and should return confidence for each detection."""
        pass

    @abstractmethod
    def calc_pairwise_ious(self, detections, labels) -> t.Dict[int, np.ndarray]:
        """Get single result from group_class_detection_label and return matrix of IOUs."""
        pass

    @abstractmethod
    def group_class_detection_label(self, detections, labels) -> t.Dict[t.Any, t.Dict[str, list]]:
        """Group detection and labels in dict of format {class_id: {'detected' [...], 'ground_truth': [...]}}."""
        pass

    @abstractmethod
    def get_detection_areas(self, detections) -> t.List[int]:
        """Get detection object of single image and should return area for each detection."""
        pass

    @abstractmethod
    def get_labels_areas(self, labels) -> t.List[int]:
        """Get labels object of single image and should return area for each label."""
        pass


class ObjectDetectionMetricMixin(MetricMixin):
    """We are expecting to receive the predictions in the following format: [x, y, w, h, confidence, label]."""

    def get_labels_areas(self, labels) -> t.List[int]:
        """Get labels object of single image and should return area for each label."""
        return [bbox[3].item() * bbox[4].item() for bbox in labels]

    def group_class_detection_label(self, detections, labels) -> t.Dict[t.Any, t.Dict[str, list]]:
        """Group detection and labels in dict of format {class_id: {'detected' [...], 'ground_truth': [...] }}."""
        return group_class_detection_label(detections, labels)

    def get_confidences(self, detections) -> t.List[float]:
        """Get detections object of single image and should return confidence for each detection."""
        return [d[4].item() for d in detections]

    def calc_pairwise_ious(self, detections, labels) -> np.ndarray:
        """Expect detection and labels of a single image and single class."""
        return compute_pairwise_ious(detections, labels, jaccard_iou)

    def get_detection_areas(self, detections) -> t.List[int]:
        """Get detection object of single image and should return area for each detection."""
        return [d[2].item() * d[3].item() for d in detections]
