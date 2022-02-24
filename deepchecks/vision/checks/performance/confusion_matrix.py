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
from typing import Any

import numpy as np
from plotly.express import imshow
from queue import PriorityQueue

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.vision import SingleDatasetCheck, Context
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.metrics_utils.iou_utils import jaccard_iou

__all__ = ['ConfusionMatrixReport']


def filter_confusion_matrix(confusion_matrix, number_of_categories):
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
    """

    def __init__(self,
                 categories_to_display: int = 10,
                 confidence_threshold: float = 0.3,
                 iou_threshold: float = 0.5):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.categories_to_display = categories_to_display
        self.iou_threshold = iou_threshold
        self.matrix = None
        self.num_classes = 0
        self.task_type = None

    def initialize_run(self, context: Context, dataset_kind: DatasetKind = None):
        """Initialize run by creating an empty matrix the size of the data."""
        context.assert_task_type(TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION)

        if dataset_kind == DatasetKind.TRAIN:
            dataset = context.train
        else:
            dataset = context.test
        self.task_type = dataset.task_type
        self.num_classes = dataset.n_of_classes

        matrix_size = self.num_classes if self.task_type == TaskType.CLASSIFICATION else self.num_classes + 1

        self.matrix = np.zeros((matrix_size, matrix_size))

    def update(self, context: Context, batch: Any, dataset_kind: DatasetKind = DatasetKind.TRAIN):
        """Add batch to confusion matrix."""
        if dataset_kind == DatasetKind.TRAIN:
            dataset = context.train
        else:
            dataset = context.test

        labels = dataset.label_formatter(batch)
        predictions = context.infer(batch)

        if self.task_type == TaskType.CLASSIFICATION:
            self.update_classification(predictions, labels)
        elif self.task_type == TaskType.OBJECT_DETECTION:
            self.update_object_detection(predictions, labels)

    def compute(self, context: Context, dataset_kind: DatasetKind = None) -> CheckResult:
        """Compute and plot confusion matrix after all batches were processed."""
        if dataset_kind == DatasetKind.TRAIN:
            dataset = context.train
        else:
            dataset = context.test
        display_confusion_matrix, categories = filter_confusion_matrix(self.matrix, self.categories_to_display)

        description = ''

        display_categories = []
        for category in categories:
            if self.num_classes == category:
                description += ('last category are detections that do not overlap with labeled data'
                                ' and labels that have not been detected. ')
                display_categories.append('not found')
            else:
                display_categories.append(dataset.label_id_to_name(category))

        description += f'Showing {self.categories_to_display} of {self.num_classes} classes:'

        fig = imshow(display_confusion_matrix,
                     x=display_categories,
                     y=display_categories,
                     text_auto=True)

        fig.update_layout(width=600, height=600)
        fig.update_xaxes(title='Predicted Value', type='category')
        fig.update_yaxes(title='True value', type='category')

        return CheckResult(
            self.matrix,
            header='Confusion Matrix',
            display=[description, fig]
        )

    def update_object_detection(self, predictions, labels):
        """Update the confusion matrix by batch for object detection task."""
        for image_detections, image_labels in zip(predictions, labels):
            detections_passed_threshold = [
                detection for detection in image_detections if detection[4] > self.confidence_threshold
            ]
            if len(detections_passed_threshold) == 0:
                # detections are empty, update matrix for labels
                for label in image_labels:
                    gt_class = int(label[0].item())
                    self.matrix[self.num_classes, gt_class] += 1
                continue

            all_ious = np.zeros((len(image_labels), len(detections_passed_threshold)))

            for label_index, label in enumerate(image_labels):
                for detected_index, detected in enumerate(detections_passed_threshold):
                    all_ious[label_index, detected_index] = jaccard_iou(detected, label)

            want_idx = np.where(all_ious > self.iou_threshold)

            all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                           for i in range(want_idx[0].shape[0])]
            all_matches = np.array(all_matches)

            # remove duplicate matches
            if all_matches.shape[0] > 0:
                all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

                all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

                all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

                all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

            for i, label in enumerate(image_labels):
                gt_class = int(label[0])
                if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                    detection_class = int(image_detections[int(all_matches[all_matches[:, 0] == i, 1][0])][5])
                    self.matrix[detection_class, gt_class] += 1
                else:
                    self.matrix[self.num_classes, gt_class] += 1

            for i, detection in enumerate(image_detections):
                if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                    detection_class = int(detection[5])
                    self.matrix[detection_class, self.num_classes] += 1

    def update_classification(self, predictions, labels):
        """Update the confusion matrix by batch for classification task."""
        for predicted_classes, image_labels in zip(predictions, labels):
            detected_class = max(range(len(predicted_classes)), key=predicted_classes.__getitem__)

            self.matrix[detected_class, image_labels] += 1
