# ----------------------------------------------------------------------------
# Copyright (C) 2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing measurements for labels and predictions."""
from typing import List
from itertools import chain

import torch

from deepchecks.core.errors import DeepchecksValueError


# Labels

def _get_bbox_area(labels: List[torch.Tensor]) -> List[int]:
    """Return a list containing the area of bboxes in batch."""
    areas = [(label.reshape((-1, 5))[:, 4] * label.reshape((-1, 5))[:, 3]).tolist()
             for label in labels]
    return list(chain.from_iterable(areas))


def _count_num_bboxes(labels: List[torch.Tensor]) -> List[int]:
    """Return a list containing the number of bboxes in per sample batch."""
    num_bboxes = [label.shape[0] for label in labels]
    return num_bboxes


def _get_samples_per_class_object_detection(labels: List[torch.Tensor]) -> List[int]:
    """Return a list containing the classes in batch."""
    classes = [tensor.reshape((-1, 5))[:, 0].tolist() for tensor in labels]
    return list(chain.from_iterable(classes))


def _get_samples_per_class_classification(labels: torch.Tensor) -> List[int]:
    """Return a list containing the class per image in batch."""
    return labels.tolist()


DEFAULT_CLASSIFICATION_LABEL_PROPERTIES = [
    {'name': 'Samples per class', 'method': _get_samples_per_class_classification, 'output_type': 'class'}
]

DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES = [
    {'name': 'Samples per class', 'method': _get_samples_per_class_object_detection, 'output_type': 'class'},
    {'name': 'Bounding box area (in pixels)', 'method': _get_bbox_area, 'output_type': 'continuous'},
    {'name': 'Number of bounding boxes per image', 'method': _count_num_bboxes, 'output_type': 'continuous'},
]


# Predictions

def _get_samples_per_predicted_class_classification(predictions: torch.Tensor) -> List[int]:
    """Return a list containing the classes in batch."""
    return torch.argmax(predictions, dim=1).tolist()


def _get_samples_per_predicted_class_object_detection(predictions: List[torch.Tensor]) -> List[int]:
    """Return a list containing the classes in batch."""
    classes = [tensor.reshape((-1, 6))[:, -1].tolist() for tensor in predictions]
    return list(chain.from_iterable(classes))


def _get_predicted_bbox_area(predictions: List[torch.Tensor]) -> List[int]:
    """Return a list containing the area of bboxes per image in batch."""
    areas = [(prediction.reshape((-1, 6))[:, 2] * prediction.reshape((-1, 6))[:, 3]).tolist()
             for prediction in predictions]
    return list(chain.from_iterable(areas))


DEFAULT_CLASSIFICATION_PREDICTION_PROPERTIES = [
    {'name': 'Samples per class', 'method': _get_samples_per_predicted_class_classification, 'output_type': 'class_id'}
]

DEFAULT_OBJECT_DETECTION_PREDICTION_PROPERTIES = [
    {'name': 'Samples per class', 'method': _get_samples_per_predicted_class_object_detection,
     'value_type': 'class_id'},
    {'name': 'Bounding box area (in pixels)', 'method': _get_predicted_bbox_area, 'output_type': 'continuous'},
    {'name': 'Number of bounding boxes per image', 'method': _count_num_bboxes, 'output_type': 'continuous'},
]


# Helper functions

def validate_properties(properties):
    """Validate structure of measurements."""
    expected_keys = ['name', 'method', 'output_type']
    output_types = ['discrete', 'continuous', 'class_id']
    if not isinstance(properties, list):
        raise DeepchecksValueError(
            f'Expected properties to be a list, instead got {properties.__class__.__name__}')
    for label_property in properties:
        if not isinstance(label_property, dict) or any(
                key not in label_property.keys() for key in expected_keys):
            raise DeepchecksValueError(f'Property must be of type dict, and include keys {expected_keys}')
        if label_property['output_type'] not in ['discrete', 'continuous']:
            raise DeepchecksValueError(f'Property field "output_type" must be one of {output_types}')
