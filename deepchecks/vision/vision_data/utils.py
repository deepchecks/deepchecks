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
import random
import sys
import typing as t
from collections import Counter
from enum import Enum
from numbers import Number

import numpy as np
from typing_extensions import NotRequired, TypedDict

from deepchecks.core.errors import DatasetValidationError
from deepchecks.utils.logger import get_logger


class TaskType(Enum):
    """Enum containing supported task types."""

    CLASSIFICATION = 'classification'
    OBJECT_DETECTION = 'object_detection'
    SEMANTIC_SEGMENTATION = 'semantic_segmentation'
    OTHER = 'other'

    @classmethod
    def values(cls):
        """Return all values of the enum."""
        return [e.value for e in TaskType]


class BatchOutputFormat(TypedDict):
    """Batch output format required by deepchecks."""

    images: NotRequired[t.Union[np.ndarray, t.Sequence]]
    labels: NotRequired[t.Union[np.ndarray, t.Sequence]]
    predictions: NotRequired[t.Union[np.ndarray, t.Sequence]]
    image_identifiers: NotRequired[t.Union[np.ndarray, t.Sequence]]


class LabelMap(dict):
    """Smarter dict for label map."""

    def __init__(self, seq=None, **kwargs):
        seq = seq or {}
        super().__init__(seq, **kwargs)

    def __getitem__(self, class_id) -> str:
        """Return the name of the class with the given id."""
        try:
            class_id = int(class_id)
        except ValueError:
            pass
        if class_id in self:
            return dict.__getitem__(self, class_id)
        return str(class_id)


def sequence_to_numpy(data: t.Optional[t.Sequence], expected_dtype=None, expected_ndim_per_object=None) -> \
        t.Optional[t.List]:
    """Convert a sequence containing some type of array to a List of numpy arrays.

    Returns
    -------
    t.Optional[t.Sequence]
        A list of numpy arrays of the provided data.
    """
    if data is None:
        return None
    return [object_to_numpy(x, expected_dtype, expected_ndim_per_object) for x in data]


def object_to_numpy(data, expected_dtype=None, expected_ndim=None) -> t.Union[np.ndarray, Number, str]:
    """Convert an object to a numpy object.

    Returns
    -------
    t.Union[np.ndarray, Number, str]
        A numpy object or a single object (number/str) for provided data.
    """
    if data is None:
        return None
    if is_torch_object(data):
        result = data.cpu().detach().numpy()
    elif is_tensorflow_object(data):
        result = data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        result = data
    elif isinstance(data, (Number, str)):
        return data
    else:
        result = np.array(data)

    if expected_dtype is not None:
        result = result.astype(expected_dtype)
    if len(result.shape) == 0:
        result = result.item()
    elif len(result.shape) == 1 and result.shape[0] > 0 and expected_ndim == 2:
        result = result.reshape(1, result.shape[0])
    return result


def shuffle_loader(batch_loader):
    """Reshuffle the batch loader."""
    if is_torch_object(batch_loader) and 'DataLoader' in str(type(batch_loader)):
        from deepchecks.vision.utils.test_utils import \
            get_data_loader_sequential  # pylint: disable=import-outside-toplevel
        try:
            _ = len(batch_loader)
            return get_data_loader_sequential(data_loader=batch_loader, shuffle=True)
        except Exception:  # pylint: disable=broad-except
            pass
    elif is_tensorflow_object(batch_loader) and 'Dataset' in str(type(batch_loader)):
        get_logger().warning('Shuffling for tensorflow datasets is not supported. Make sure that the data used to '
                             'create the Dataset was shuffled beforehand and set shuffle_batch_loader=False')
        return batch_loader
    get_logger().warning('Shuffling is not supported for received batch loader. Make sure that your provided '
                         'batch loader is indeed shuffled and set shuffle_batch_loader=False')
    return batch_loader


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
        return Counter(np.hstack(class_ids_per_image)) if len(class_ids_per_image) > 0 else {}
    elif task_type == TaskType.SEMANTIC_SEGMENTATION:
        labels_per_image = [np.unique(label) for label in labels if label is not None]
        return Counter(np.hstack(labels_per_image))
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
        classes_predicted_per_image = \
            [np.unique(np.argmax(pred, axis=0)) for pred in predictions if pred is not None]
        return Counter(np.hstack(classes_predicted_per_image))
    else:
        raise ValueError(f'Unsupported task type: {task_type}')


def is_torch_object(data_object) -> bool:
    """Check if data_object is a torch object without failing if torch isn't installed."""
    return 'torch' in str(type(data_object))


def is_tensorflow_object(data_object) -> bool:
    """Check if data_object is a tensorflow object without failing if tensorflow isn't installed."""
    return 'tensorflow' in str(type(data_object))


def set_seeds(seed: int):
    """Set seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed to be set
    """
    if seed is not None and isinstance(seed, int):
        np.random.seed(seed)
        random.seed(seed)
        if 'torch' in sys.modules:
            import torch  # pylint: disable=import-outside-toplevel
            torch.manual_seed(seed)
        if 'tensorflow' in sys.modules:
            import tensorflow as tf  # pylint: disable=import-outside-toplevel
            tf.random.set_seed(seed)


def validate_vision_data_compatibility(first, second) -> None:
    """Validate that two vision datasets are compatible.

    Raises:
        DeepchecksValueError: if the datasets are not compatible
    """
    # TODO: add more validations
    if first.task_type != second.task_type:
        raise DatasetValidationError('Cannot compare datasets with different task types: '
                                     f'{first.task_type.value} and {second.task_type.value}')
