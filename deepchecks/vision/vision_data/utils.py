# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Utils module for VisionData functionalities."""
import typing as t
from collections import Counter
from enum import Enum
from numbers import Number

import numpy as np
import torch


class TaskType(Enum):
    """Enum containing supported task types."""

    CLASSIFICATION = 'classification'
    OBJECT_DETECTION = 'object_detection'
    SEMANTIC_SEGMENTATION = 'semantic_segmentation'
    OTHER = 'other'


class BatchOutputFormat(t.TypedDict):
    images: t.Optional[t.Union[np.ndarray, t.Sequence]]
    labels: t.Optional[t.Union[np.ndarray, t.Sequence]]
    predictions: t.Optional[t.Union[np.ndarray, t.Sequence]]
    additional_data: t.Optional[t.Union[np.ndarray, t.Sequence]]
    embeddings: t.Optional[t.Union[np.ndarray, t.Sequence]]
    image_identifiers: t.Optional[t.Union[np.ndarray, t.Sequence]]


def sequence_to_numpy(data: t.Optional[t.Sequence], expected_dtype=None) -> t.Optional[t.Sequence]:
    """Convert a sequence containing some type of array to a List of numpy arrays.

    Returns
    -------
    t.Optional[t.Sequence]
        A list of numpy arrays of the provided data.
    """
    if data is None:
        return None
    return [object_to_numpy(x, expected_dtype) for x in data]


def object_to_numpy(data, expected_dtype=None) -> t.Union[np.ndarray, Number, str]:
    """Convert an object to a numpy object.

    Returns
    -------
    t.Union[np.ndarray, Number, str]
        A numpy object or a single object (number/str) for provided data.
    """
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        result = data.numpy()
    elif isinstance(data, np.ndarray):
        result = data
    elif isinstance(data, Number) or isinstance(data, str):
        return data
    else:
        raise ValueError(f'Unsupported data type: {type(data)}')

    if expected_dtype is not None:
        result = result.astype(expected_dtype)
    if len(result.shape) == 0:
        result = result.item()
    return result


def shuffle_dynamic_loader(dynamic_loader):
    """Reshuffle the dynamic loader."""
    # TODO: do something here
    return dynamic_loader


def get_class_ids_from_numpy_labels(labels: t.Sequence[t.Union[np.ndarray, int]], task_type: TaskType) \
        -> t.Dict[int, int]:
    """Return the number of images containing each class_id.

    Returns
    -------
    Dict[int, int]
        A dictionary mapping each class_id to the number of images containing it.
    """
    if task_type == TaskType.CLASSIFICATION:
        return Counter(labels)
    elif task_type == TaskType.OBJECT_DETECTION:
        class_ids_per_image = [label[:, 0] for label in labels if label is not None and len(label.shape) == 2]
        return Counter(np.hstack(class_ids_per_image))
    elif task_type == TaskType.SEMANTIC_SEGMENTATION:
        return 4
    else:
        raise ValueError(f'Unsupported task type: {task_type}')


def get_class_ids_from_numpy_preds(predictions: t.Sequence[t.Union[np.ndarray]], task_type: TaskType) \
        -> t.Dict[int, int]:
    """Return the number of images containing each class_id.

    Returns
    -------
    Dict[int, int]
        A dictionary mapping each class_id to the number of images containing it.
    """
    if task_type == TaskType.CLASSIFICATION:
        return Counter([np.argmax(x) for x in predictions])
    elif task_type == TaskType.OBJECT_DETECTION:
        class_ids_per_image = [pred[:, 5] for pred in predictions if pred is not None and len(pred.shape) == 2]
        return Counter(np.hstack(class_ids_per_image))
    elif task_type == TaskType.SEMANTIC_SEGMENTATION:
        return 4
    else:
        raise ValueError(f'Unsupported task type: {task_type}')

