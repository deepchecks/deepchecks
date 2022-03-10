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
from typing import List, Iterable

from deepchecks.core import DatasetKind
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import VisionData, Context


# Labels

def _get_bbox_area(label, _):
    """Return a list containing the area of bboxes per image in batch."""
    areas = (label.reshape((-1, 5))[:, 4] * label.reshape((-1, 5))[:, 3]).reshape(-1, 1).tolist()
    return areas


def _count_num_bboxes(label, _):
    """Return a list containing the number of bboxes per image in batch."""
    num_bboxes = label.shape[0]
    return num_bboxes


def _get_samples_per_class_classification(label, dataset):
    """Return a list containing the class per image in batch."""
    return dataset.label_id_to_name(label.tolist())


def _get_samples_per_class_object_detection(label, dataset):
    """Return a list containing the class per image in batch."""
    return [[dataset.label_id_to_name(arr.reshape((-1, 5))[:, 0])] for arr in label]


DEFAULT_CLASSIFICATION_LABEL_MEASUREMENTS = [
    {'name': 'Samples per class', 'method': _get_samples_per_class_classification, 'is_continuous': False}
]

DEFAULT_OBJECT_DETECTION_LABEL_MEASUREMENTS = [
    {'name': 'Samples per class', 'method': _get_samples_per_class_object_detection, 'is_continuous': False},
    {'name': 'Bounding box area (in pixels)', 'method': _get_bbox_area, 'is_continuous': True},
    {'name': 'Number of bounding boxes per image', 'method': _count_num_bboxes, 'is_continuous': True},
]


# Predictions

def _get_samples_per_predicted_class_classification(prediction, dataset):
    """Return a list containing the class per image in batch."""
    return dataset.label_id_to_name(prediction.argmax().tolist())


def _get_samples_per_predicted_class_object_detection(prediction, dataset):
    """Return a list containing the class per image in batch."""
    return [[dataset.label_id_to_name(arr.reshape((-1, 6))[:, -1])] for arr in prediction]


def _get_predicted_bbox_area(prediction, _):
    """Return a list containing the area of bboxes per image in batch."""
    areas = (prediction.reshape((-1, 6))[:, 2] * prediction.reshape((-1, 6))[:, 3]).reshape(-1, 1).tolist()
    return areas


DEFAULT_CLASSIFICATION_PREDICTION_MEASUREMENTS = [
    {'name': 'Samples per class', 'method': _get_samples_per_predicted_class_classification, 'is_continuous': False}
]

DEFAULT_OBJECT_DETECTION_PREDICTION_MEASUREMENTS = [
    {'name': 'Samples per class', 'method': _get_samples_per_predicted_class_object_detection, 'is_continuous': False},
    {'name': 'Bounding box area (in pixels)', 'method': _get_predicted_bbox_area, 'is_continuous': True},
    {'name': 'Number of bounding boxes per image', 'method': _count_num_bboxes, 'is_continuous': True},
]


# Helper functions

def get_label_measurements_on_batch(batch, label_measurement, dataset: VisionData):
    """Calculate transformer result on batch of labels."""
    calc_res = [label_measurement(arr, dataset) for arr in batch.labels]
    return flatten(calc_res)


def get_prediction_measurements_on_batch(batch, prediction_measurement, dataset, context: Context,
                                         dataset_kind: DatasetKind):
    """Calculate transformer result on batch of labels."""
    calc_res = [prediction_measurement(arr, dataset) for arr in batch.predictions]
    return flatten(calc_res)


def flatten(in_list: List) -> List:
    """Flatten a list of lists (nested infinitely) into a single level list."""

    def inner_flatten(inner_list: Iterable) -> List:
        for el in inner_list:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from inner_flatten(el)
            else:
                yield el

    return list(inner_flatten(in_list))


def validate_measurements(measurements):
    """Validate structure of measurements."""
    expected_keys = ['name', 'method', 'is_continuous']
    if not isinstance(measurements, list):
        raise DeepchecksValueError(
            f'Expected measurements to be a list, instead got {measurements.__class__.__name__}')
    for label_measurement in measurements:
        if not isinstance(label_measurement, dict) or any(
                key not in label_measurement.keys() for key in expected_keys):
            raise DeepchecksValueError(f'Measurement must be of type dict, and include keys {expected_keys}')
