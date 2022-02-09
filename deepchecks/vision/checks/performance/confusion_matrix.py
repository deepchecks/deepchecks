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
from typing import Union

import numpy as np
import plotly.express as px
from torch import nn
from queue import PriorityQueue

from deepchecks.core import CheckResult
from deepchecks.vision import SingleDatasetCheck, Context
from deepchecks.vision.dataset import TaskType, VisionData
from deepchecks.vision.metrics_utils.iou_utils import _jaccard
from deepchecks.vision.utils import ClassificationPredictionFormatter, DetectionPredictionFormatter


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

    Parameters
    ----------
    confidence_threshold (float, default 0.3):
        Threshold to consider object as detected.
    categories_to_display (int, default 10):
        Maximum number of categories to display
    confidence_threshold (float, default 0.3):
        Threshold to consider bounding box as detected.
    iou_threshold (float, default 0.5):
        Threshold to consider detected bounding box as labeled bounding box.
    """

    def __init__(self,
                 prediction_formatter: Union[ClassificationPredictionFormatter, DetectionPredictionFormatter] = None,
                 categories_to_display: int = 10,
                 confidence_threshold: float = 0.3,
                 iou_threshold: float = 0.5):
        super().__init__()
        self.prediction_formatter = prediction_formatter
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.categories_to_display = categories_to_display

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is the full confusion matrix.
        """
        dataset: VisionData = context.train

        model: nn.Module = context.model
        context.assert_task_type(TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION)

        calculator = CalculateConfusionMatrix(dataset.get_num_classes(), self.categories_to_display, self.confidence_threshold, self.iou_threshold)

        for images, labels in dataset.get_data_loader():
            labels = dataset.label_transformer(labels)

            predictions = model.forward(images)

            if self.prediction_formatter:
                predictions = self.prediction_formatter(predictions)

            calculator.process_batch(predictions, labels)

        return calculator.compute_display()


class CalculateConfusionMatrix:
    """Calculate the confusion matrix on batches.

    """
    def __init__(self, num_classes: int, categories_to_display: int = 10, conf_threshold=0.3, iou_threshold=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.categories_to_display = categories_to_display
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def process_batch(self, detections, labels: np.ndarray):
        """Add batch to confusion matrix.

        """
        for image_detections, image_labels in zip(detections, labels):
            try:
                detections_passed_threshold = [
                    detection for detection in image_detections if detection[4] > self.conf_threshold
                ]
            except IndexError:
                # detections are empty, update matrix for labels
                for label in image_labels:
                    gt_class = label[0]
                    self.matrix[self.num_classes, gt_class] += 1
                continue

            all_ious = np.zeros((len(image_labels), len(detections_passed_threshold)))

            for label_index, label in enumerate(image_labels):
                for detected_index, detected in enumerate(detections_passed_threshold):
                    all_ious[label_index, detected_index] = _jaccard(detected, label)

            want_idx = np.where(all_ious > self.iou_threshold)

            all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                           for i in range(want_idx[0].shape[0])]
            all_matches = np.array(all_matches)

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

    def compute_display(self) -> CheckResult:
        display_confusion_matrix, categories = filter_confusion_matrix(self.matrix, self.categories_to_display)

        fig = px.imshow(display_confusion_matrix,
                        x=categories,
                        y=categories,
                        text_auto=True)
        #fig = px.imshow(self.matrix,
        #                x=list(range(self.num_classes+1)),
        #                y=list(range(self.num_classes+1)),
        #                text_auto=True)
        fig.update_layout(width=600, height=600)
        fig.update_xaxes(title='Predicted Value', type='category')
        fig.update_yaxes(title='True value', type='category')

        return CheckResult(
            self.matrix,
            header='Confusion Matrix',
            display=fig
        )
