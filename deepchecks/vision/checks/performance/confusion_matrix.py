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
"""Module containing performance report check."""
from collections import defaultdict
from typing import Callable, TypeVar, List, Union

import numpy as np
import plotly.express as px
from torch import nn

from deepchecks.core import CheckResult, ConditionResult
from deepchecks.vision import SingleDatasetCheck, Context
from deepchecks.vision.dataset import TaskType, VisionData
from deepchecks.vision.metrics_utils.iou_utils import compute_ious, _jaccard

__all__ = ['ConfusionMatrixReport']

from deepchecks.vision.utils import ClassificationPredictionFormatter, DetectionPredictionFormatter

PR = TypeVar('PR', bound='PerformanceReport')


class ConfusionMatrixReport(SingleDatasetCheck):
    """Calculate the confusion matrix of the model on the given dataset.

    Parameters
    ----------
    confidence_threshold (float, default 0.3):
        Threshold to consider object as detected.
    iou_threshold (float, default 0.3):
        Threshold to consider object as detected.
    """

    def __init__(self,
                 prediction_formatter: Union[ClassificationPredictionFormatter, DetectionPredictionFormatter] = None,
                 confidence_threshold: float = 0.3, iou_threshold: float = 0.5):
        super().__init__()
        self.prediction_formatter = prediction_formatter
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is dictionary in format 'score-name': score-value
        """
        #if dataset_type == 'train':
        dataset: VisionData = context.train
        #else:
        #    dataset: VisionData = context.test

        model: nn.Module = context.model
        context.assert_task_type(TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION)

        calculator = CalculateConfusionMatrix(dataset.get_num_classes(), self.confidence_threshold, self.iou_threshold)

        for images, labels in dataset.get_data_loader():
            labels = dataset.label_transformer(labels)

            predictions = model.forward(images)

            if self.prediction_formatter:
                predictions = self.prediction_formatter(predictions)

            calculator.process_batch(predictions, labels)

        confusion_matrix = calculator.return_matrix()

        print(confusion_matrix)
        print(confusion_matrix.shape)

        fig = px.imshow(confusion_matrix, x=list(range(0,dataset.get_num_classes()+1)), y=list(range(dataset.get_num_classes()+1)), text_auto=True)
        fig.update_layout(width=600, height=600)
        fig.update_xaxes(title='Predicted Value', type='category')
        fig.update_yaxes(title='True value', type='category')

        return CheckResult(
            confusion_matrix,
            header='Confusion Matrix',
            display=fig
        )


class CalculateConfusionMatrix:
    """Calculate the confusion matrix on batches


    """
    def __init__(self, num_classes: int, conf_threshold=0.3, iou_threshold=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def process_batch(self, detections, labels: np.ndarray):
        """Add batch to confusion matrix

        """
        for image_detections, image_labels in zip(detections, labels):
            try:
                detections_passed_threshold = [
                    detection for detection in image_detections if detection[4] > self.conf_threshold
                ]
            except IndexError or TypeError:
                # detections are empty, end of process
                for label in image_labels:
                    gt_class = label[0]
                    self.matrix[self.num_classes, gt_class] += 1
                return

            all_ious = np.zeros((len(image_labels), len(detections_passed_threshold)))

            for label_index, label in enumerate(image_labels):
                for detected_index, detected in enumerate(detections_passed_threshold):
                    all_ious[label_index, detected_index] = _jaccard(detected, label)

            want_idx = np.where(all_ious > self.iou_threshold)

            all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                           for i in range(want_idx[0].shape[0])]
            all_matches = np.array(all_matches)

            if all_matches.shape[0] > 0:  # if there is match
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

    def return_matrix(self):
        return self.matrix
