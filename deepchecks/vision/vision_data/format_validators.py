# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module validating the VisionData functionalities implemented by the user."""
from numbers import Number
from typing import Iterable

import numpy as np

from deepchecks.core.errors import ValidationError
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.vision_data.utils import object_to_numpy, sequence_to_numpy


def validate_images_format(images):
    """Validate that the data is in the required format.

    Parameters
    ----------
    images
        The images of the batch. Result of VisionData's batch_to_images
    Raises
    ------
    DeepchecksValueError
        If the images doesn't fit the required deepchecks format.
    """
    try:
        image = object_to_numpy(images[0], expected_ndim=3)
    except TypeError as err:
        raise ValidationError(f'The batch images must be an iterable, received {type(images)}.') from err
    try:
        if len(image.shape) != 3:
            raise ValidationError('The image inside the iterable must be a 3D array.')
    except Exception as err:
        raise ValidationError('The image inside the iterable must be a 3D array.') from err
    if image.shape[2] not in [1, 3]:
        raise ValidationError('The image inside the iterable must have 1 or 3 channels.')
    sample_min = np.min(image)
    sample_max = np.max(image)
    if sample_min < 0 or sample_max > 255 or sample_max <= 1:
        raise ValidationError(f'Image data should be in uint8 format(integers between 0 and 255), '
                              f'found values in range [{sample_min}, {sample_max}].')


def validate_labels_format(labels, task_type: TaskType):
    """Validate that the labels are in the required format based on task_type.

    Parameters
    ----------
    labels
        The labels of the batch. Result of VisionData's batch_to_labels
    task_type: TaskType
        The task type of the model
    Raises
    ------
    DeepchecksValueError
        If the labels doesn't fit the required deepchecks format.
    """
    single_image_label = _validate_predictions_label_common_format('label', labels, task_type)

    if task_type == TaskType.CLASSIFICATION and single_image_label is not None:
        if not isinstance(single_image_label, Number):
            raise ValidationError('Classification label per image must be a number.')
    elif task_type == TaskType.OBJECT_DETECTION and single_image_label is not None:
        error_msg = 'Object detection label per image must be a sequence of 2D arrays, where each row ' \
                    'has 5 columns: [class_id, x_min, y_min, width, height]'
        if single_image_label.shape[1] != 5:
            raise ValidationError(f'{error_msg}')
        if np.min(single_image_label) < 0:
            raise ValidationError(f'Found one of coordinates to be negative, {error_msg}')
        if np.max(single_image_label[:, 0] % 1) > 0:
            raise ValidationError(f'Class_id must be a positive integer. {error_msg}')
    elif task_type == TaskType.SEMANTIC_SEGMENTATION and single_image_label is not None:
        if len(single_image_label.shape) != 2:
            raise ValidationError('Semantic segmentation label per image must be a 2D array of shape (H, W),'
                                  'where H and W are the height and width of the corresponding image.')
        if np.max(single_image_label % 1) > 0:
            raise ValidationError('In semantic segmentation, each pixel in label should represent a '
                                  'class_id and there must be a positive integer.')


def validate_predictions_format(predictions, task_type: TaskType):
    """Validate that the predictions are in the required format based on task_type.

    Parameters
    ----------
    predictions
        The predictions of the batch. Result of VisionData's batch_to_predictions
    task_type: TaskType
        The task type of the model
    Raises
    ------
    DeepchecksValueError
        If the predictions doesn't fit the required deepchecks format.
    """
    single_image_pred = _validate_predictions_label_common_format('prediction', predictions, task_type)

    if task_type == TaskType.CLASSIFICATION and single_image_pred is not None:
        if not isinstance(single_image_pred, np.ndarray) or not isinstance(single_image_pred[0], Number) or \
                not 0.99 < np.sum(single_image_pred) < 1.01:
            raise ValidationError('Classification prediction per image must be a sequence of floats representing '
                                  'probabilities per class.')
    elif task_type == TaskType.OBJECT_DETECTION and single_image_pred is not None:
        error_msg = 'Object detection prediction per image must be a sequence of 2D arrays, where each row ' \
                    'has 6 columns: [x_min, y_min, w, h, confidence, class_id]'
        if single_image_pred.shape[1] != 6:
            raise ValidationError(error_msg)
        if np.min(single_image_pred) < 0:
            raise ValidationError(f'Found one of coordinates to be negative, {error_msg}')
        if np.max(single_image_pred[:, 5] % 1) > 0:
            raise ValidationError(f'Class_id must be a positive integer. {error_msg}')
    elif task_type == TaskType.SEMANTIC_SEGMENTATION and single_image_pred is not None:
        if len(single_image_pred.shape) != 3:
            raise ValidationError('Semantic segmentation prediction per image must be a 3D array of shape (C, H, W),'
                                  'where H and W are the height and width of the corresponding image, and C is the '
                                  'number of classes that can be detected.')
        if not 0.99 < np.sum(single_image_pred[:, 0][:, 0]) < 1.01:
            raise ValidationError('Semantic segmentation prediction per pixel per image should be probabilities per '
                                  'each possible class')


def _validate_predictions_label_common_format(name, data, task_type: TaskType):
    """Validate that the data is in the required format and returns a non-empty sample in numpy format."""
    name_plural = name + 's'
    if task_type == TaskType.OTHER:
        return None
    try:
        _ = data[0]
        data = sequence_to_numpy(data)
    except (IndexError, KeyError, TypeError) as err:
        raise ValidationError(f'The batch {name_plural} must be a non empty iterable.') from err

    sample_idx = 0
    while data[sample_idx] is None or (isinstance(data[sample_idx], np.ndarray) and data[sample_idx].shape[0] == 0):
        sample_idx += 1
        if sample_idx == len(data):
            return None  # No data to validate

    if task_type == TaskType.CLASSIFICATION:
        return data[sample_idx]  # for classification, the data is a single number no need to validate shape
    try:
        sample_shape = data[sample_idx].shape
    except AttributeError as err:
        raise ValidationError(f'{name} for {task_type.value} per image must be a multi dimensional array.') from err

    if task_type == TaskType.OBJECT_DETECTION and len(sample_shape) not in (1, 2):
        raise ValidationError(f'{name} for object detection per image must be a 2D array. Found shape {sample_shape}')

    return data[sample_idx]


def validate_additional_data_format(additional_data_batch):
    """Validate that the data is in the required format.

    Parameters
    ----------
    additional_data_batch
        The additional data of the batch. Result of VisionData's batch_to_additional_data
    Raises
    ------
    DeepchecksValueError
        If the images doesn't fit the required deepchecks format.
    """
    if not isinstance(additional_data_batch, Iterable):
        raise ValidationError('The batch additional_data must be an iterable.')


def validate_embeddings_format(embeddings):
    """Validate that the data is in the required format.

    Parameters
    ----------
    embeddings
        The embeddings of the batch. Result of VisionData's batch_to_embeddings
    Raises
    ------
    DeepchecksValueError
        If the images doesn't fit the required deepchecks format.
    """
    if not isinstance(embeddings, Iterable):
        raise ValidationError('The batch embeddings must be an iterable.')


def validate_image_identifiers_format(image_identifiers):
    """Validate that the data is in the required format.

    Parameters
    ----------
    image_identifiers
        The image identifiers of the batch. Result of VisionData's batch_to_image_identifiers
    Raises
    ------
    DeepchecksValueError
        If the images doesn't fit the required deepchecks format.
    """
    try:
        sample = image_identifiers[0]
    except TypeError as err:
        raise ValidationError('The batch image_identifiers must be an iterable.') from err
    if not isinstance(sample, str):
        raise ValidationError('The image identifier inside the iterable must be a string.')
