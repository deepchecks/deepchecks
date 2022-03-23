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
from functools import partial
from itertools import product
from textwrap import dedent

import pandas as pd
import numpy as np
import torch
from plotly.express import imshow
from queue import PriorityQueue
from collections import defaultdict

from deepchecks.vision.utils.image_functions import prepare_thumbnail
from deepchecks.vision.utils.image_functions import draw_bboxes
from deepchecks.core import CheckResult, DatasetKind
from deepchecks.vision import SingleDatasetCheck, Context, Batch
from deepchecks.vision.vision_data import TaskType, VisionData
from deepchecks.vision.metrics_utils.iou_utils import jaccard_iou


__all__ = ['ConfusionMatrixReport']


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
    n_of_images_to_show : int, default 5
        Number of misclassified images to show.
    """

    _IMAGE_THUMBNAIL_SIZE = (400, 400)
    _LABEL_COLOR = 'red'
    _DETECTION_COLOR = 'blue'

    def __init__(
        self,
        categories_to_display: int = 10,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        n_of_images_to_show: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.categories_to_display = categories_to_display
        self.iou_threshold = iou_threshold
        self.matrix = None
        self.task_type = None
        self.misclassified_images = None
        self.n_of_images_to_show = n_of_images_to_show

    def initialize_run(self, context: Context, dataset_kind: DatasetKind):
        """Initialize run by creating an empty matrix the size of the data."""
        context.assert_task_type(TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION)
        dataset = context.get_data_by_kind(dataset_kind)
        self.task_type = dataset.task_type
        self.matrix = defaultdict(lambda: defaultdict(int))
        self.misclassified_images = []

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind = DatasetKind.TRAIN):
        """Add batch to confusion matrix."""
        if self.task_type == TaskType.CLASSIFICATION:
            self.update_classification(batch)
        elif self.task_type == TaskType.OBJECT_DETECTION:
            self.update_object_detection(batch)

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Compute and plot confusion matrix after all batches were processed."""
        assert self.matrix is not None

        dataset = context.get_data_by_kind(dataset_kind)
        matrix = pd.DataFrame(self.matrix).T
        matrix.replace(np.nan, 0, inplace=True)

        classes = sorted(
            set(matrix.index).union(set(matrix.columns)),
            key=lambda x: np.inf if isinstance(x, str) else x
        )

        matrix = pd.DataFrame(matrix, index=classes, columns=classes)

        confusion_matrix, categories = filter_confusion_matrix(
            matrix.to_numpy(),
            self.categories_to_display
        )

        description = [f'Showing {self.categories_to_display} of {dataset.num_classes} classes:']
        classes_to_display_ids = []
        classes_to_display = []
        classes_map = dict(enumerate(classes))  # class index -> class label

        for category in categories:
            category = classes_map[category]
            if category == 'no-overlapping':
                description.append(
                    '"No overlapping" categories are labels and prediction which did not have a matching '
                    'label/prediction. For example a predictions that did not have a sufficiently overlapping '
                    'label bounding box will appear under the "No overlapping label" category'
                )
                classes_to_display_ids.append('no-overlapping')
                classes_to_display.append('no-overlapping')
            elif isinstance(category, int):
                classes_to_display_ids.append(category)
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
                x.append('No overlapping prediction')
                y.append('No overlapping label')

        description.append(
            imshow(
                confusion_matrix,
                x=x,
                y=y,
                text_auto=True)
            .update_layout(width=600, height=600)
            .update_xaxes(title='Predicted Value', type='category')
            .update_yaxes(title='True value', type='category')
        )
        description.append(self._misclassified_images_thumbnails(
            matrix.loc[classes_to_display_ids, classes_to_display_ids],
            dataset
        ))

        del self.misclassified_images
        del self.matrix

        return CheckResult(
            matrix,
            header='Confusion Matrix',
            display=description
        )

    def update_object_detection(self, batch: Batch):
        """Update the confusion matrix by batch for object detection task."""
        assert self.matrix is not None
        assert self.misclassified_images is not None

        labels = batch.labels
        predictions = batch.predictions
        images = batch.images

        for image, detected_bboxes, label_bboxes in zip(images, predictions, labels):
            detections_passed_threshold = [
                detection for detection in detected_bboxes
                if detection[4] > self.confidence_threshold
            ]

            if len(detections_passed_threshold) == 0:
                # detections are empty, update matrix for labels
                for label in label_bboxes:
                    label_class = int(label[0].item())
                    self.matrix[label_class]['no-overlapping'] += 1
                continue

            list_of_ious = (
                (label_index, detected_index, jaccard_iou(detected, label))
                for label_index, label in enumerate(label_bboxes)
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

            for index, bbox in enumerate(label_bboxes):
                bbox_class = int(bbox[0])
                if n_of_matches > 0 and (matches[:, 0] == index).any():
                    detection_index = int(matches[matches[:, 0] == index, 1][0])
                    detected_bbox = detected_bboxes[detection_index]
                    detected_class = int(detected_bbox[5])
                    self.matrix[bbox_class][detected_class] += 1
                    if bbox_class != detected_class:
                        # NOTE:
                        # not all misclassified images will be displayed
                        # therefore to omit unneeded work at current stage
                        # we will wrap bbox drawing step into callable
                        img = partial(
                            self._draw_bboxes,
                            image=image,
                            label=np.array([
                                bbox.numpy()
                                if isinstance(bbox, torch.Tensor)
                                else bbox]),
                            detected=np.array([
                                detected_bbox.numpy()
                                if isinstance(detected_bbox, torch.Tensor)
                                else detected_bbox])
                        )
                        self.misclassified_images.append((bbox_class, detected_class, img))
                else:
                    self.matrix[bbox_class]['no-overlapping'] += 1

            for index, detected_bbox in enumerate(detections_passed_threshold):
                if n_of_matches > 0 and not (matches[:, 1] == index).any():
                    detected_class = int(detected_bbox[5])
                    self.matrix['no-overlapping'][detected_class] += 1

    def update_classification(self, batch: Batch):
        """Update the confusion matrix by batch for classification task."""
        assert self.matrix is not None
        assert self.misclassified_images is not None

        labels = batch.labels
        predictions = batch.predictions
        images = batch.images

        for image, predicted_classes, image_labels in zip(images, predictions, labels):
            detected_class = max(range(len(predicted_classes)), key=predicted_classes.__getitem__)
            label_class = image_labels.item() if isinstance(image_labels, torch.Tensor) else image_labels
            self.matrix[label_class][detected_class] += 1
            if label_class != detected_class:
                self.misclassified_images.append((label_class, detected_class, image))

    def _misclassified_images_thumbnails(
        self,
        matrix: pd.DataFrame,
        dataset: VisionData
    ) -> str:
        assert self.misclassified_images is not None

        if self.n_of_images_to_show == 0:
            return ''

        template = dedent(f"""
            <h4>Misclassified Images</h4>
            <p><b>NOTE:</b> in case of object detection task the next colors
            are used for bbox identification:</p>
            <ul>
                <li>{self._LABEL_COLOR} - ground truth</li>
                <li>{self._DETECTION_COLOR} - detected truth</li>
            </ul>
            <p>Showing {self.n_of_images_to_show} of {{n_of_images}} images:</p>
            <div
                style="
                    overflow-x: auto;
                    display: grid;
                    grid-template-rows: auto;
                    grid-template-columns: auto auto 1fr;
                    grid-gap: 1.5rem;
                    justify-items: center;
                    align-items: center;
                    padding: 2rem;
                    width: max-content;">
                {{data}}
            </div>
        """)

        misclassifications = (
            (
                label_class,
                detected_class,
                matrix.at[label_class, detected_class]  # count
            )
            for label_class, detected_class in product(list(matrix.index), repeat=2)
            if label_class != detected_class
        )
        misclassifications = sorted(
            misclassifications,
            key=lambda it: it[2],
            reverse=True
        )
        misclassified_images = [
            (
                (dataset.label_id_to_name(x_label), x_label),
                (dataset.label_id_to_name(x_detected), x_detected),
                img
            )
            for x_label, x_detected, _ in misclassifications
            for (y_label, y_detected, img) in self.misclassified_images
            if x_label == y_label and x_detected == y_detected
        ]

        if len(misclassified_images) == 0:
            return ''

        grid = [
            '<span><b>Ground Truth</b></span>',
            '<span><b>Detected Truth</b></span>',
            '<span><b>Image</b></span>',
        ]

        for label, detected, img in misclassified_images[:self.n_of_images_to_show]:
            label_name, label_id = label
            detection_name, detection_id = detected
            grid.append(f'<span>{label_name} (id: {label_id})</span>')
            grid.append(f'<span>{detection_name} (id: {detection_id})</span>')
            # NOTE: take a look at the update_object_detection method
            # to understand why 'img' might be a callable
            grid.append(prepare_thumbnail(
                image=img() if callable(img) else img,
                size=self._IMAGE_THUMBNAIL_SIZE
            ))

        return template.format(
            n_of_images=len(misclassified_images),
            data=''.join(grid)
        )

    def _draw_bboxes(self, image, label, detected):
        img = draw_bboxes(
            image=image,
            bboxes=label,
            border_width=2,
            color=self._LABEL_COLOR
        )
        return draw_bboxes(
            image=img,
            bboxes=detected,
            bbox_notation='xywhl',
            border_width=2,
            color=self._DETECTION_COLOR,
            copy_image=False
        )


def filter_confusion_matrix(
    confusion_matrix: np.ndarray,
    number_of_categories: int
) -> t.Tuple[np.ndarray, t.List[int]]:
    pq = PriorityQueue()
    for row_index, column_index, value in flatten_matrix(confusion_matrix):
        if row_index != column_index:
            pq.put((-value, (row_index, column_index)))
    categories = set()
    while not pq.empty():
        if len(categories) >= number_of_categories:
            break
        _, (row, col) = pq.get()
        categories.add(row)
        categories.add(col)
    categories = sorted(categories)
    return confusion_matrix[np.ix_(categories, categories)], categories


def flatten_matrix(matrix: np.ndarray) -> t.Iterator[t.Tuple[int, int, t.Any]]:
    for row_index, row in enumerate(matrix):
        for column_index, cell in enumerate(row):
            yield row_index, column_index, cell
