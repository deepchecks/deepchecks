# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains Train Test label Drift check."""
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import plotly.graph_objs as go
import torch
from plotly.subplots import make_subplots

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision import Batch, Context, TrainTestCheck
from deepchecks.vision.utils.image_functions import apply_heatmap_image_properties, numpy_grayscale_to_heatmap_figure
from deepchecks.vision.vision_data import TaskType, VisionData

__all__ = ['HeatmapComparison']


class HeatmapComparison(TrainTestCheck):
    """Check if the average image brightness (or bbox location if applicable) is similar between train and test set.

    The check computes the average grayscale image per dataset (train and test) and compares the resulting images.
    Additionally, in case of an object detection task, the check will compare the average locations of the bounding
    boxes between the datasets.

    Parameters
    ----------
    classes_to_display : Optional[List[float]], default: None
        List of classes to display in bounding box heatmap, using the class names (strings). Applies only for
        object detection tasks. If None, all classes are displayed.
    """

    def __init__(self,
                 classes_to_display: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.classes_to_display = {str(x) for x in classes_to_display} if classes_to_display else None

    def initialize_run(self, context: Context):
        """Initialize run.

        Function initializes the following private variables:

        * self._task_type: TaskType
        * self._train_grayscale_heatmap: variable aggregating the average training grayscale image
        * self._test_grayscale_heatmap: variable aggregating the average test grayscale image
        * self._shape: List containing the target image shape (determined by the first image encountered)
        * self._train_counter: Number of training images aggregated
        * self._test_counter: Number of test images aggregated

        If task is object detection, we also initialize the following variables:

        * self._train_bbox_heatmap: variable aggregating the average training bounding box heatmap
        * self._test_bbox_heatmap: variable aggregating the average test bounding box heatmap
        """
        train_dataset = context.train

        self._task_type = train_dataset.task_type
        self._class_to_string = context.train.label_id_to_name

        # State members to store the average grayscale image throughout update steps
        self._grayscale_heatmap = defaultdict(lambda: 0)
        self._shape = None
        self._counter = {}
        self._counter[DatasetKind.TRAIN] = 0
        self._counter[DatasetKind.TEST] = 0

        # State members to store the average bounding box heatmap throughout update steps
        if self._task_type == TaskType.OBJECT_DETECTION:
            self._bbox_heatmap = defaultdict(lambda: 0)

    def update(self, context: Context, batch: Batch, dataset_kind):
        """Perform update on batch for train or test counters and histograms."""
        # For each dataset kind, update the respective counter and histogram with the average image of the batch
        # The counter accumulates the number of images in the batches
        # image_batch is a list of images, each of which is a numpy array of shape (H, W, C), produced by running
        # the image_formatter on the batch.
        data = context.get_data_by_kind(dataset_kind)
        valid_labels, valid_images = self._filter_images(data, batch)
        if len(valid_images) != 0:
            self._counter[dataset_kind] += len(valid_images)
            summed_image = self._grayscale_sum_image(valid_images)
            self._grayscale_heatmap[dataset_kind] += summed_image
            # For object detection tasks, we do the same for the bounding box average coverage of the image.
            # The difference from the above code for the average grayscale image is
            # that the averaged images are images of
            # the places where the bounding boxes are located. These bounding box images are computed by
            # _label_to_image_batch
            if self._task_type == TaskType.OBJECT_DETECTION:
                label_image_batch = self._label_to_image_batch(valid_labels, valid_images)
                summed_bbox_image = self._grayscale_sum_image(label_image_batch)
                self._bbox_heatmap[dataset_kind] += summed_bbox_image

    def compute(self, context: Context) -> CheckResult:
        """Create the average images and display them.

        Returns
        -------
        CheckResult
            value: The difference images. One for average image brightness, and one for bbox locations if applicable.
            display: Heatmaps for image brightness (train, test, diff) and heatmap for bbox locations if applicable.
        """
        # if self.classes_to_display is set, check that it has classes that actually exist
        if self.classes_to_display is not None:
            if not self.classes_to_display.issubset(
                    map(self._class_to_string, context.train.classes_indices.keys())
            ):
                raise DeepchecksValueError(
                    f'Provided list of class ids to display {list(self.classes_to_display)} not found in training '
                    f'dataset.'
                )
        # Compute the average grayscale image by dividing the accumulated sum by the number of images
        train_grayscale = (np.expand_dims(self._grayscale_heatmap[DatasetKind.TRAIN], axis=2) /
                           self._counter[DatasetKind.TRAIN]).astype(np.uint8)
        test_grayscale = (np.expand_dims(self._grayscale_heatmap[DatasetKind.TEST], axis=2) /
                          self._counter[DatasetKind.TEST]).astype(np.uint8)

        value = {
            'diff': self._image_diff(test_grayscale, train_grayscale)
        }

        if context.with_display:
            # Add a display for the heatmap of the average grayscale image
            display = [self.plot_row_of_heatmaps(train_grayscale, test_grayscale, 'Compare average image brightness')]
            display[0].update_layout(coloraxis={'colorscale': 'Inferno', 'cmin': 0, 'cmax': 255},
                                     coloraxis_colorbar={'title': 'Pixel Value'})

        # If the task is object detection, compute the average heatmap of the bounding box locations by dividing the
        # accumulated sum by the number of images
        if self._task_type == TaskType.OBJECT_DETECTION:
            # bbox image values are frequency, between 0 and 100
            train_bbox = (100 * np.expand_dims(self._bbox_heatmap[DatasetKind.TRAIN], axis=2) /
                          self._counter[DatasetKind.TRAIN] / 255).astype(np.uint8)
            test_bbox = (100 * np.expand_dims(self._bbox_heatmap[DatasetKind.TEST], axis=2) /
                         self._counter[DatasetKind.TEST] / 255).astype(np.uint8)

            value['diff_bbox'] = self._image_diff(test_bbox, train_bbox)

            if context.with_display:
                display.append(
                    self.plot_row_of_heatmaps(train_bbox, test_bbox, 'Compare average label bbox locations')
                )
                # bbox image values are frequency, between 0 and 100
                display[1].update_layout(coloraxis={'colorscale': 'Inferno', 'cmin': 0, 'cmax': 100},
                                         coloraxis_colorbar={'title': '% Coverage'})

        return CheckResult(value=value,
                           display=display if context.with_display else None,
                           header='Heatmap Comparison')

    @staticmethod
    def plot_row_of_heatmaps(train_img: np.ndarray, test_img: np.ndarray, title: str) -> go.Figure:
        """Plot a row of heatmaps for train and test images."""
        fig = make_subplots(rows=1, cols=2, column_titles=['Train', 'Test'])
        fig.add_trace(numpy_grayscale_to_heatmap_figure(train_img), row=1, col=1)
        fig.add_trace(numpy_grayscale_to_heatmap_figure(test_img), row=1, col=2)
        fig.update_yaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
        fig.update_xaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
        fig.update_layout(title=title)
        apply_heatmap_image_properties(fig)
        return fig

    @staticmethod
    def _image_diff(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Return the difference between two grayscale images as a grayscale image."""
        diff = img1.astype(np.int32) - img2.astype(np.int32)
        return np.abs(diff).astype(np.uint8)

    def _filter_images(self, data: VisionData, batch: Batch) -> \
            Tuple[List[torch.Tensor], List[np.ndarray]]:
        """Filter the images by the classes to display and return the valid labels and images."""
        valid_images = []
        valid_labels = []

        if self.classes_to_display is None:
            valid_images.extend(batch.images)
            valid_labels.extend(batch.labels)
        else:
            samples = batch.images, batch.labels, data.get_classes(batch.labels)
            for image, label, classes in zip(*samples):
                class_names = {self._class_to_string(c) for c in classes}
                if len(class_names.intersection(self.classes_to_display)) > 0:
                    valid_labels.append(label)
                    valid_images.append(image)

        return valid_labels, valid_images

    def _label_to_image(self, label: np.ndarray, original_shape: Tuple[int]) -> np.ndarray:
        """Convert label array to an image where pixels inside the bboxes are white and the rest are black."""
        # Create a black image
        image = np.zeros(original_shape, dtype=np.uint8)
        label = label.reshape((-1, 5))
        class_idx = label[:, 0]
        x_min = (label[:, 1]).astype(np.int32)
        y_min = (label[:, 2]).astype(np.int32)
        x_max = (label[:, 1] + label[:, 3]).astype(np.int32)
        y_max = (label[:, 2] + label[:, 4]).astype(np.int32)
        for i in range(len(label)):
            # If classes_to_display is set, don't display the bboxes for classes not in the list.
            if self.classes_to_display is not None and \
               self._class_to_string(class_idx[i]) not in self.classes_to_display:
                continue
            # For each bounding box, set the pixels inside the bounding box to white
            image[y_min[i]:y_max[i], x_min[i]:x_max[i]] = 255
        return np.expand_dims(image, axis=2)

    def _label_to_image_batch(self, label_batch: List[torch.Tensor], image_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Convert label batch to batch of images where pixels inside the bboxes are white and the rest are black."""
        return_bbox_image_batch = []
        for image, label in zip(image_batch, label_batch):
            return_bbox_image_batch.append(
                self._label_to_image(label.cpu().detach().numpy(), image.shape[:2])
            )
        return return_bbox_image_batch

    def _grayscale_sum_image(self, batch: Iterable[np.ndarray]) -> np.ndarray:
        """Sum all images in batch to one grayscale image of shape target_shape.

        Parameters
        ----------
        batch: np.ndarray
            batch of images.
        target_shape: List[Tuple[int, int]], default: None
            list containing shape of image. If empty, the shape is taken from the first image in the batch.

        Returns
        -------
        np.ndarray
            summed image.
        """
        summed_image = None

        # Iterate over all images in batch, using the first image as the target shape if target_shape is None.
        # All subsequent images will be resized to the target shape and their gray values will be added to the
        # summed image.
        for img in batch:
            # Cast to grayscale
            if img.shape[2] == 1:
                resized_img = img
            elif img.shape[2] == 3:
                resized_img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY)
            else:
                raise NotImplementedError('Images must be RGB or grayscale')

            # reshape to one shape
            if self._shape is None:
                self._shape = resized_img.shape[:2][::-1]
            resized_img = cv2.resize(resized_img.astype('uint8'), self._shape, interpolation=cv2.INTER_AREA)

            # sum images
            if summed_image is None:
                summed_image = resized_img.squeeze().astype(np.int64)
            else:
                summed_image += resized_img.squeeze().astype(np.int64)

        return summed_image
