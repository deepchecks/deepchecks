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
from abc import abstractmethod
from typing import Dict, List

import numpy as np

class MetricMixin:
    @abstractmethod
    def get_confidences(self, detections) -> List[float]:
        """Get detections object of single image and should return confidence for each detection."""
        pass

    @abstractmethod
    def calc_pairwise_ious(self, detections, labels) -> Dict[int, np.ndarray]:
        """Get single result from group_class_detection_label and return matrix of IOUs."""
        pass

    @abstractmethod
    def group_class_detection_label(self, detections, labels) -> dict:
        """Group detection and labels in dict of format {class_id: {'detected' [...], 'ground_truth': [...]}}."""
        pass

    @abstractmethod
    def get_detection_areas(self, detections) -> List[int]:
        """Get detection object of single image and should return area for each detection."""
        pass

    @abstractmethod
    def get_labels_areas(self, labels) -> List[int]:
        """Get labels object of single image and should return area for each label."""
        pass
