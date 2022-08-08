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
from typing import List, Sequence

import torch

# Labels


def _get_bbox_area(labels: List[torch.Tensor]) -> List[List[int]]:
    """Return a list containing the area of bboxes in batch."""
    return [(label.reshape((-1, 5))[:, 4] * label.reshape((-1, 5))[:, 3]).tolist()
            for label in labels]


def _count_num_bboxes(labels: List[torch.Tensor]) -> List[int]:
    """Return a list containing the number of bboxes in per sample batch."""
    num_bboxes = [label.shape[0] for label in labels]
    return num_bboxes


def _get_samples_per_class_object_detection(labels: List[torch.Tensor]) -> List[List[int]]:
    """Return a list containing the classes in batch."""
    return [tensor.reshape((-1, 5))[:, 0].tolist() for tensor in labels]


def _get_samples_per_class_classification(labels: torch.Tensor) -> List[int]:
    """Return a list containing the class per image in batch."""
    return labels.tolist()


DEFAULT_CLASSIFICATION_LABEL_PROPERTIES = [
    {'name': 'Samples Per Class', 'method': _get_samples_per_class_classification, 'output_type': 'class_id'}
]

DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES = [
    {'name': 'Samples Per Class', 'method': _get_samples_per_class_object_detection, 'output_type': 'class_id'},
    {'name': 'Bounding Box Area (in pixels)', 'method': _get_bbox_area, 'output_type': 'numerical'},
    {'name': 'Number of Bounding Boxes Per Image', 'method': _count_num_bboxes, 'output_type': 'numerical'},
]


# Predictions

def _get_samples_per_predicted_class_classification(predictions: torch.Tensor) -> List[int]:
    """Return a list containing the classes in batch."""
    return torch.argmax(predictions, dim=1).tolist()


def _get_samples_per_predicted_class_object_detection(predictions: List[torch.Tensor]) -> List[List[int]]:
    """Return a list containing the classes in batch."""
    return [tensor.reshape((-1, 6))[:, -1].tolist() for tensor in predictions]


def _get_predicted_bbox_area(predictions: List[torch.Tensor]) -> List[List[int]]:
    """Return a list containing the area of bboxes per image in batch."""
    return [(prediction.reshape((-1, 6))[:, 2] * prediction.reshape((-1, 6))[:, 3]).tolist()
            for prediction in predictions]


DEFAULT_CLASSIFICATION_PREDICTION_PROPERTIES = [
    {
        'name': 'Samples Per Class',
        'method': _get_samples_per_predicted_class_classification,
        'output_type': 'class_id'
    }
]

DEFAULT_OBJECT_DETECTION_PREDICTION_PROPERTIES = [
    {
        'name': 'Samples Per Class',
        'method': _get_samples_per_predicted_class_object_detection,
        'output_type': 'class_id'
    },
    {
        'name': 'Bounding Box Area (in pixels)',
        'method': _get_predicted_bbox_area,
        'output_type': 'numerical'},
    {
        'name': 'Number of Bounding Boxes Per Image',
        'method': _count_num_bboxes,
        'output_type': 'numerical'
    },
]


# Helper functions
def get_column_type(output_type):
    """Get column type to use in drift functions."""
    # TODO smarter mapping based on data?
    # NOTE/TODO: this function is kept only for backward compatibility, remove it later
    mapper = {
        'continuous': 'numerical',
        'discrete': 'categorical',
        'class_id': 'categorical',
        'numerical': 'numerical',
        'categorical': 'categorical',
    }
    return mapper[output_type]


def properties_flatten(in_list: Sequence) -> List:
    """Flatten a list of lists into a single level list."""
    out = []
    for el in in_list:
        if isinstance(el, Sequence) and not isinstance(el, (str, bytes)):
            out.extend(el)
        else:
            out.append(el)
    return out
