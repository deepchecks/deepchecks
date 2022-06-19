# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing confusion matrix report check."""
import typing as t
from collections import defaultdict
from queue import PriorityQueue

import numpy as np
import pandas as pd
import torch

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.utils.plot import create_confusion_matrix_figure
from deepchecks.vision import Batch, Context, SingleDatasetCheck
from deepchecks.vision.metrics_utils.iou_utils import jaccard_iou
from deepchecks.vision.vision_data import TaskType

__all__ = ['ConfusionMatrixReport']


def filter_confusion_matrix(confusion_matrix: pd.DataFrame, number_of_categories: int) -> \
                            t.Tuple[np.ndarray, int]:
    pq = PriorityQueue()
    for row, values in enumerate(confusion_matrix):
        for col, value in enumerate(values):
            if row != col:
                pq.put((-value, (row, col)))
    categories = set()
    while not pq.empty():
        if len(categories) >= number_of_categories:
            break
        _, (row, col) = pq.get()
        categories.add(row)
        categories.add(col)
    categories = sorted(categories)
    return confusion_matrix[np.ix_(categories, categories)], categories


class ConfusionMatrixReport(SingleDatasetCheck):
    """Calculate the confusion matrix of the model on the given dataset.

    For object detection, each detected bounding box calculates the IoU for each label and then is that label class is
    used for the confusion matrix. detected bounding boxes that don't match a label has their own class and same
    for labels without detected bounding boxes.

    Parameters
    ----------
    categories_to_display (int, default 10):
        Maximum number of categories to display
    confidence_threshold (float, default 0.3):
        Threshold to consider bounding box as detected.
    iou_threshold (float, default 0.5):
        Threshold to consider detected bounding box as labeled bounding box.
    normalized (bool, default True):
        boolean that determines whether to normalize the true values of the matrix.
    """

    def __init__(self,
                 categories_to_display: int = 10,
                 confidence_threshold: float = 0.3,
                 iou_threshold: float = 0.5,
                 normalized: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.categories_to_display = categories_to_display
        self.iou_threshold = iou_threshold
        self.normalized = normalized

        self.matrix = None
        self.classes_list = None
        self.not_found_idx = None
        self.unseen_class_idx = None
        self.task_type = None

    def initialize_run(self, context: Context, dataset_kind: DatasetKind = None):
        """Initialize run by creating an empty matrix the size of the data."""
        context.assert_task_type(TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION)
        dataset = context.get_data_by_kind(dataset_kind)
        self.task_type = dataset.task_type
        self.matrix = defaultdict(lambda: defaultdict(int))

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind = DatasetKind.TRAIN):
        """Add batch to confusion matrix."""
        labels = batch.labels
        predictions = batch.predictions
        if self.task_type == TaskType.CLASSIFICATION:
            self.update_classification(predictions, labels)
        elif self.task_type == TaskType.OBJECT_DETECTION:
            self.update_object_detection(predictions, labels)

    def compute(self, context: Context, dataset_kind: DatasetKind = None) -> CheckResult:
        """Compute and plot confusion matrix after all batches were processed."""
        assert self.matrix is not None

        dataset = context.get_data_by_kind(dataset_kind)
        matrix = pd.DataFrame(self.matrix).T
        matrix = matrix.rename(index={-1: 'no-overlapping'}, columns={-1: 'no-overlapping'})
        matrix.replace(np.nan, 0, inplace=True)

        classes = sorted(
            set(matrix.index).union(set(matrix.columns)),
            key=lambda x: np.inf if isinstance(x, str) else x
        )

        matrix = pd.DataFrame(matrix, index=classes, columns=classes).to_numpy()

        if context.with_display:
            confusion_matrix, categories = filter_confusion_matrix(
                matrix,
                self.categories_to_display
            )
            confusion_matrix = np.nan_to_num(confusion_matrix)

            description = [f'Showing {self.categories_to_display} of {dataset.num_classes} classes:']
            classes_to_display = []
            classes_map = dict(enumerate(classes))  # class index -> class label

            for category in categories:
                category = classes_map[category]
                if category == 'no-overlapping':
                    description.append(
                        '"No overlapping" categories are labels and prediction which did not have a matching '
                        'label/prediction.<br>For example a predictions that did not have a sufficiently overlapping '
                        'label bounding box will appear under "No overlapping" category in the True Value '
                        'axis (y-axis).'
                    )
                    classes_to_display.append('no-overlapping')
                elif isinstance(category, int):
                    classes_to_display.append(dataset.label_id_to_name(category))
                else:
                    raise RuntimeError(
                        'Internal Error! categories list must '
                        'contain items of type - Union[int, Literal["no-overlapping"]]'
                    )

            x = []
            y = []

            for it in classes_to_display:
                if it != 'no-overlapping':
                    x.append(it)
                    y.append(it)
                else:
                    x.append('No overlapping')
                    y.append('No overlapping')

            description.append(
                create_confusion_matrix_figure(confusion_matrix, x, y, self.normalized)
            )
        else:
            description = None

        return CheckResult(
            matrix,
            header='Confusion Matrix',
            display=description
        )

    def update_object_detection(self, predictions, labels):
        """Update the confusion matrix by batch for object detection task."""
        assert self.matrix is not None

        for image_detections, image_labels in zip(predictions, labels):
            detections_passed_threshold = [
                detection for detection in image_detections
                if detection[4] > self.confidence_threshold
            ]

            if len(detections_passed_threshold) == 0:
                # detections are empty, update matrix for labels
                for label in image_labels:
                    label_class = int(label[0].item())
                    self.matrix[label_class][-1] += 1
                continue

            list_of_ious = (
                (label_index, detected_index, jaccard_iou(detected.cpu().detach().numpy(),
                                                          label.cpu().detach().numpy()))
                for label_index, label in enumerate(image_labels)
                for detected_index, detected in enumerate(detections_passed_threshold)
            )
            matches = np.array([
                [label_index, detected_index, ious]
                for label_index, detected_index, ious in list_of_ious
                if ious > self.iou_threshold
            ])

            # remove duplicate matches
            if len(matches) > 0:
                # sort by ious, in descend order
                matches = matches[matches[:, 2].argsort()[::-1]]
                # leave matches with unique prediction and the highest ious
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # sort by ious, in descend order
                matches = matches[matches[:, 2].argsort()[::-1]]
                # leave matches with unique label and the highest ious
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            n_of_matches = len(matches)

            for label_index, label in enumerate(image_labels):
                label_class = int(label[0])
                if n_of_matches > 0 and (matches[:, 0] == label_index).any():
                    detection_index = int(matches[matches[:, 0] == label_index, 1][0])
                    detected_class = int(image_detections[detection_index][5])
                    self.matrix[label_class][detected_class] += 1
                else:
                    self.matrix[label_class][-1] += 1

            for detection_index, detection in enumerate(detections_passed_threshold):
                if n_of_matches > 0 and not (matches[:, 1] == detection_index).any():
                    detected_class = int(detection[5])
                    self.matrix[-1][detected_class] += 1

    def update_classification(self, predictions, labels):
        """Update the confusion matrix by batch for classification task."""
        assert self.matrix is not None

        for predicted_classes, image_labels in zip(predictions, labels):
            detected_class = max(range(len(predicted_classes)), key=predicted_classes.__getitem__)
            label_class = image_labels.item() if isinstance(image_labels, torch.Tensor) else image_labels
            self.matrix[label_class][detected_class] += 1
