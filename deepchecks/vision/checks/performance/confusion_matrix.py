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
import pandas as pd
import numpy as np
from plotly.express import imshow
from queue import PriorityQueue

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.vision import SingleDatasetCheck, Context, Batch
from deepchecks.vision.vision_data import TaskType
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


def getat(df, cell, default = None):
    row, column = cell
    if row in df.index and column in df.columns:
        return df.at[cell]
    else:
        return default


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
        self.classes_list = None
        self.not_found_idx = None
        self.unseen_class_idx = None
        self.task_type = None

    def initialize_run(self, context: Context, dataset_kind: DatasetKind = None):
        """Initialize run by creating an empty matrix the size of the data."""
        context.assert_task_type(TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION)
        dataset = context.train if dataset_kind == DatasetKind.TRAIN else context.test
        
        self.task_type = dataset.task_type
        self.matrix = pd.DataFrame()

        # # In case of object detection add last category for "not found label/prediction overlap"
        # if self.task_type == TaskType.OBJECT_DETECTION:
        #     # In detection, we might have non-consecutive ids. For example, we might have class ids 10, 14, 20. So we
        #     # will use the class list as a map of matrix id to class id: 0: 10, 1: 14, 2: 20
        #     self.classes_list = sorted([int(x) for x in dataset.n_of_samples_per_class.keys()])
        #     # Adding 2 extra categories. One for predictions with unseen before classes, and second for label and
        #     # prediction that have no overlapping prediction/label
        #     matrix_size = len(self.classes_list) + 2
        #     self.not_found_idx = matrix_size + 1
        #     self.unseen_class_idx = matrix_size + 2
        # else:
        #     matrix_size = dataset.num_classes

        # self.matrix = np.zeros((matrix_size, matrix_size))

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
        
        dataset = context.train if dataset_kind == DatasetKind.TRAIN else context.test
        
        classes = list(set(self.matrix.index).union(set(self.matrix.columns)))
        classes_map = dict(enumerate(classes)) # class index -> class
        matrix = pd.DataFrame(self.matrix, index=classes, columns=classes)
        matrix.replace(np.nan, 0, inplace=True)

        confusion_matrix, categories = filter_confusion_matrix(
            matrix.to_numpy(), 
            self.categories_to_display
        )

        description = [f'Showing {self.categories_to_display} of {dataset.num_classes} classes:']
        classes_to_display = []
        # add_not_found_category = False
        # add_unseen_category = False
        
        for category in categories:
            category = classes_map[category]
            if category == 'no-overlapping':
                description.append(
                    '"No overlapping" categories are labels and prediction which did not have a matching '
                    'label/prediction. For example a predictions that did not have a sufficiently overlapping '
                    'label bounding box will appear under the "No overlapping label" category'
                )
                classes_to_display.append('no-overlapping')
            else:
                classes_to_display.append(dataset.label_id_to_name(category))

        # x = display_categories
        # y = x.copy()
        # description = []

        # if add_not_found_category:
        #     description += ['"No overlapping" categories are labels and prediction which did not have a matching '
        #                     'label/prediction. For example a predictions that did not have a sufficiently overlapping '
        #                     'label bounding box will appear under the "No overlapping label" category']
        #     x += ['No overlapping prediction']
        #     y += ['No overlapping label']
        # if add_unseen_category:
        #     description += ['Unseen classes are classes that did not appear in the dataset labels, but did were '
        #                     'predicted to be present.']
        #     x += ['Unseen classes']
        #     y += ['Unseen classes']

        # description += [f'Showing {self.categories_to_display} of {dataset.num_classes} classes:']

        fig = (
            imshow(
                confusion_matrix,
                x=classes_to_display,
                y=classes_to_display,
                text_auto=True)
            .update_layout(width=600, height=600)
            .update_xaxes(title='Predicted Value', type='category')
            .update_yaxes(title='True value', type='category')
        )
        return CheckResult(
            self.matrix,
            header='Confusion Matrix',
            display=[*description, fig]
        )

    # def class_id_to_matrix_id(self, class_id):
    #     """Convert a class id to its id in the matrix."""
    #     try:
    #         return self.classes_list.index(class_id)
    #     # If the class is not in the labels list (detection returned class not in the list) then return not found
    #     except ValueError:
    #         return self.unseen_class_idx

    # def matrix_id_to_class_name(self, matrix_id, dataset):
    #     """Convert matrix id to the real class id, and return its name if possible."""
    #     class_id = self.classes_list[matrix_id] if self.classes_list else matrix_id
    #     return dataset.label_id_to_name(class_id)

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
                    self.matrix.at[label_class, 'no-overlapping'] = getat(
                        self.matrix, 
                        (label_class, 'no-overlapping'), 
                        default=0
                    ) + 1
                continue
            
            list_of_ious = (
                (label_index, detected_index, jaccard_iou(detected, label))
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
                    self.matrix.at[label_class, detected_class] = getat(
                        self.matrix, 
                        (label_class, detected_class), 
                        default=0
                    ) + 1
                else:
                    self.matrix.at[label_class, 'no-overlapping'] = getat(
                        self.matrix, 
                        (label_class, 'no-overlapping'), 
                        default=0
                    ) + 1

            for detection_index, detection in enumerate(detections_passed_threshold):
                if n_of_matches > 0 and not (matches[:, 1] == detection_index).any():
                    detected_class = int(detection[5])
                    self.matrix.at['no-overlapping', detected_class] = getat(
                        self.matrix, 
                        ('no-overlapping', detected_class), 
                        default=0
                    ) + 1

    def update_classification(self, predictions, labels):
        """Update the confusion matrix by batch for classification task."""
        assert self.matrix is not None
        
        for predicted_classes, image_labels in zip(predictions, labels):
            detected_class = max(range(len(predicted_classes)), key=predicted_classes.__getitem__)
            # self.matrix[image_labels, detected_class] += 1
            self.matrix.at[image_labels, detected_class] = getat(
                self.matrix, 
                (image_labels, detected_class), 
                default=0
            ) + 1
