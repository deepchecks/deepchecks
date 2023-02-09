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
from typing import List, Sequence, Union

import numpy as np

# Labels


def _get_samples_per_class_classification(labels: Union[np.ndarray, List]) -> List[int]:
    """Return a list containing the class per image in batch."""
    return labels if isinstance(labels, List) else labels.tolist()


def _get_samples_per_class_object_detection(labels: List[np.ndarray]) -> List[List[int]]:
    """Return a list containing the classes in batch."""
    return [tensor.reshape((-1, 5))[:, 0].tolist() for tensor in labels]


def _get_bbox_area(labels: List[np.ndarray]) -> List[List[int]]:
    """Return a list containing the area of bboxes in batch."""
    return [(label.reshape((-1, 5))[:, 4] * label.reshape((-1, 5))[:, 3]).tolist()
            for label in labels]


def _count_num_bboxes(labels: List[np.ndarray]) -> List[int]:
    """Return a list containing the number of bboxes in per sample batch."""
    num_bboxes = [label.shape[0] for label in labels]
    return num_bboxes


def _get_samples_per_class_semantic_segmentation(labels: List[np.ndarray]) -> List[List[int]]:
    """Return a list containing the classes in batch."""
    return [np.unique(label).tolist() for label in labels]


def _get_segment_area(labels: List[np.ndarray]) -> List[List[int]]:
    """Return a list containing the area of segments in batch."""
    return [np.unique(label, return_counts=True)[1].tolist() for label in labels]


def _count_classes_by_segment_in_image(labels: List[np.ndarray]) -> List[int]:
    """Return a list containing the number of unique classes per image for semantic segmentation."""
    return [np.unique(label).shape[0] for label in labels]


DEFAULT_CLASSIFICATION_LABEL_PROPERTIES = [
    {'name': 'Samples Per Class', 'method': _get_samples_per_class_classification, 'output_type': 'class_id'}
]

DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES = [
    {'name': 'Samples Per Class', 'method': _get_samples_per_class_object_detection, 'output_type': 'class_id'},
    {'name': 'Bounding Box Area (in pixels)', 'method': _get_bbox_area, 'output_type': 'numerical'},
    {'name': 'Number of Bounding Boxes Per Image', 'method': _count_num_bboxes, 'output_type': 'numerical'},
]

DEFAULT_SEMANTIC_SEGMENTATION_LABEL_PROPERTIES = [
    {'name': 'Samples Per Class', 'method': _get_samples_per_class_semantic_segmentation, 'output_type': 'class_id'},
    {'name': 'Segment Area (in pixels)', 'method': _get_segment_area, 'output_type': 'numerical'},
    {'name': 'Number of Classes Per Image', 'method': _count_classes_by_segment_in_image, 'output_type': 'numerical'},
]


# Predictions

def _get_predicted_classes_per_image_classification(predictions: List[np.ndarray]) -> List[int]:
    """Return a list of the predicted class per image in the batch."""
    return np.argmax(predictions, axis=1).tolist()


def _get_predicted_classes_per_image_object_detection(predictions: List[np.ndarray]) -> List[List[int]]:
    """Return a list containing the classes in batch."""
    return [bboxes_per_image.reshape((-1, 6))[:, -1].tolist() for bboxes_per_image in predictions]


def _get_predicted_bbox_area(predictions: List[np.ndarray]) -> List[List[int]]:
    """Return a list of the predicted bbox sizes per image in the batch."""
    return [(prediction.reshape((-1, 6))[:, 2] * prediction.reshape((-1, 6))[:, 3]).tolist()
            for prediction in predictions]


def _get_predicted_classes_per_image_semantic_segmentation(predictions: List[np.ndarray]) -> List[List[int]]:
    """Return a list containing the classes in batch."""
    return [np.unique(pred.argmax(0)).tolist() for pred in predictions]


def _get_segment_pred_area(predictions: List[np.ndarray]) -> List[List[int]]:
    """Return a list containing the area of segments in batch."""
    return [np.unique(pred.argmax(0), return_counts=True)[1].tolist() for pred in predictions]


def _count_pred_classes_by_segment_in_image(predictions: List[np.ndarray]) -> List[int]:
    """Return a list containing the number of unique classes per image for semantic segmentation."""
    return [np.unique(preds.argmax(0)).shape[0] for preds in predictions]


DEFAULT_CLASSIFICATION_PREDICTION_PROPERTIES = [
    {'name': 'Samples Per Class', 'method': _get_predicted_classes_per_image_classification, 'output_type': 'class_id'}
]

DEFAULT_OBJECT_DETECTION_PREDICTION_PROPERTIES = [
    {'name': 'Samples Per Class', 'method': _get_predicted_classes_per_image_object_detection,
     'output_type': 'class_id'},
    {'name': 'Bounding Box Area (in pixels)', 'method': _get_predicted_bbox_area, 'output_type': 'numerical'},
    {'name': 'Number of Bounding Boxes Per Image', 'method': _count_num_bboxes, 'output_type': 'numerical'},
]

DEFAULT_SEMANTIC_SEGMENTATION_PREDICTION_PROPERTIES = [
    {'name': 'Samples Per Class', 'method': _get_predicted_classes_per_image_semantic_segmentation,
     'output_type': 'class_id'},
    {'name': 'Segment Area (in pixels)', 'method': _get_segment_pred_area, 'output_type': 'numerical'},
    {'name': 'Number of Classes Per Image', 'method': _count_pred_classes_by_segment_in_image,
     'output_type': 'numerical'},
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
