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
import warnings
from typing import Any, Dict, List, Sequence

import torch

from deepchecks.core.errors import DeepchecksValueError

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

def validate_properties(properties: List[Dict[str, Any]]):
    """Validate structure of measurements."""
    if not isinstance(properties, list):
        raise DeepchecksValueError(
            'Expected properties to be a list, '
            f'instead got {type(properties).__name__}'
        )

    if len(properties) == 0:
        raise DeepchecksValueError('Properties list can\'t be empty')

    expected_keys = ('name', 'method', 'output_type')
    deprecated_output_types = ('discrete', 'continuous')
    output_types = ('categorical', 'numerical', 'class_id')

    errors = []
    list_of_warnings = []

    for index, label_property in enumerate(properties):

        if not isinstance(label_property, dict):
            errors.append(
                f'Item #{index}: property must be of type dict, '
                f'and include keys {expected_keys}. Instead got {type(label_property).__name__}'
            )
            continue

        property_name = label_property.get('name') or f'#{index}'
        difference = sorted(set(expected_keys).difference(set(label_property.keys())))

        if len(difference) > 0:
            errors.append(
                f'Property {property_name}: dictionary must include keys {expected_keys}. '
                f'Next keys are missed {difference}'
            )
            continue

        property_output_type = label_property['output_type']

        if property_output_type in deprecated_output_types:
            list_of_warnings.append(
                f'Property {property_name}: output types {deprecated_output_types} are deprecated, '
                f'use instead {output_types}'
            )
        elif property_output_type not in output_types:
            errors.append(
                f'Property {property_name}: field "output_type" must be one of {output_types}, '
                f'instead got {property_output_type}'
            )

    if len(errors) > 0:
        errors = '\n+ '.join(errors)
        raise DeepchecksValueError(f'List of properties contains next problems:\n+ {errors}')

    if len(list_of_warnings) > 0:
        concatenated_warnings = '\n+ '.join(list_of_warnings)
        warnings.warn(
            f'Property Warnings:\n+ {concatenated_warnings}',
            category=DeprecationWarning
        )

    return properties


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
