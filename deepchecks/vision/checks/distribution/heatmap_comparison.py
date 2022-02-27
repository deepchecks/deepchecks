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
from typing import Tuple, List, Any, Iterable, Optional

import cv2
import torch
from plotly.subplots import make_subplots

from deepchecks.vision.utils.image_functions import numpy_grayscale_to_heatmap_figure, apply_heatmap_image_properties

from deepchecks.core import DatasetKind, CheckResult
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.vision.base import Context, TrainTestCheck
from deepchecks.vision.dataset import TaskType
import numpy as np
import plotly.graph_objs as go

__all__ = ['HeatmapComparison']


class HeatmapComparison(TrainTestCheck):
    """
    Check if the average image brightness (or bbox location if applicable) is similar between train and test set.

    The check computes the average grayscale image per dataset (train and test) and compares the resulting images.
    This comparison may serve to visualize differences in the statistics of the datasets. Additionally, in case of an
    object detection task, the check will compare the average locations of the bounding boxes between the datasets.

    Parameters
    ----------
    classes_to_display : Optional[List[float]], default: None
        List of classes to display in bounding box heatmap, using the class names (strings). Applies only for
        object detection tasks. If None, all classes are displayed.
    """

    def __init__(self,
                 classes_to_display: Optional[List[str]] = None):
        super().__init__()
        self.classes_to_display = classes_to_display

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

        # if self.classes_to_display is set, check that it has classes that actually exist
        if self.classes_to_display is not None:
            if not self._task_type == TaskType.OBJECT_DETECTION:
                raise DeepchecksNotSupportedError('Classes to display is only supported for object detection tasks.')
            if not set(self.classes_to_display).issubset(
                    map(self._class_to_string, train_dataset.n_of_samples_per_class.keys())
            ):
                raise DeepchecksValueError(
                    f'Provided list of class ids to display {self.classes_to_display} not found in training dataset.'
                )

        # State members to store the average grayscale image throughout update steps
        self._train_grayscale_heatmap = None
        self._test_grayscale_heatmap = None
        self._shape = []
        self._train_counter = 0
        self._test_counter = 0

        # State members to store the average bounding box heatmap throughout update steps
        if self._task_type == TaskType.OBJECT_DETECTION:
            self._train_bbox_heatmap = None
            self._test_bbox_heatmap = None

    def update(self, context: Context, batch: Any, dataset_kind):
        """Perform update on batch for train or test counters and histograms."""
        # For each dataset kind, update the respective counter and histogram with the average image of the batch
        # The counter accumulates the number of images in the batches
        # image_batch is a list of images, each of which is a numpy array of shape (H, W, C), produced by running
        # the image_formatter on the batch.
        if dataset_kind == DatasetKind.TRAIN:
            image_batch = context.train.image_formatter(batch)
            summed_image = self._grayscale_sum_image(image_batch, self._shape)
            if self._train_grayscale_heatmap is None:
                self._train_grayscale_heatmap = summed_image
            else:
                self._train_grayscale_heatmap += summed_image
            self._train_counter += len(image_batch)
        elif dataset_kind == DatasetKind.TEST:
            image_batch = context.test.image_formatter(batch)
            summed_image = self._grayscale_sum_image(image_batch, self._shape)
            if self._test_grayscale_heatmap is None:
                self._test_grayscale_heatmap = summed_image
            else:
                self._test_grayscale_heatmap += summed_image
            self._test_counter += len(image_batch)
        else:
            raise DeepchecksNotSupportedError(f'Unsupported dataset kind {dataset_kind}')

        # For object detection tasks, we do the same for the bounding box average coverage of the image.
        # The difference from the above code for the average grayscale image is that the averaged images are images of
        # the places where the bounding boxes are located. These bounding box images are computed by
        # _label_to_image_batch
        if self._task_type == TaskType.OBJECT_DETECTION:
            if dataset_kind == DatasetKind.TRAIN:
                label_batch = context.train.label_formatter(batch)
                label_image_batch = self._label_to_image_batch(label_batch, image_batch, self.classes_to_display)
                summed_image = self._grayscale_sum_image(label_image_batch, self._shape)
                if self._train_bbox_heatmap is None:
                    self._train_bbox_heatmap = summed_image
                else:
                    self._train_bbox_heatmap += summed_image
            elif dataset_kind == DatasetKind.TEST:
                label_batch = context.test.label_formatter(batch)
                label_image_batch = self._label_to_image_batch(label_batch, image_batch, self.classes_to_display)
                summed_image = self._grayscale_sum_image(label_image_batch, self._shape)
                if self._test_bbox_heatmap is None:
                    self._test_bbox_heatmap = summed_image
                else:
                    self._test_bbox_heatmap += summed_image

    def compute(self, context: Context) -> CheckResult:
        """Create the average images and display them.

        Returns
        -------
        CheckResult
            value: The difference images. One for average image brightness, and one for bbox locations if applicable.
            display: Heatmaps for image brightness (train, test, diff) and heatmap for bbox locations if applicable.
        """
        # Compute the average grayscale image by dividing the accumulated sum by the number of images
        train_grayscale = (np.expand_dims(self._train_grayscale_heatmap, axis=2) /
                           self._train_counter).astype(np.uint8)
        test_grayscale = (np.expand_dims(self._test_grayscale_heatmap, axis=2) /
                          self._test_counter).astype(np.uint8)

        # Add a display for the heatmap of the average grayscale image
        display = [self.plot_row_of_heatmaps(train_grayscale, test_grayscale, 'Compare average image brightness')]
        display[0].update_layout(coloraxis={'colorscale': 'Inferno', 'cmin': 0, 'cmax': 255},
                                 coloraxis_colorbar={'title': 'Pixel Value'})
        value = {
            'diff': self._image_diff(test_grayscale, train_grayscale)
        }

        # If the task is object detection, compute the average heatmap of the bounding box locations by dividing the
        # accumulated sum by the number of images
        if self._task_type == TaskType.OBJECT_DETECTION:
            # bbox image values are frequency, between 0 and 100
            train_bbox = (100 * np.expand_dims(self._train_bbox_heatmap, axis=2) /
                          self._train_counter / 255).astype(np.uint8)
            test_bbox = (100 * np.expand_dims(self._test_bbox_heatmap, axis=2) /
                         self._test_counter / 255).astype(np.uint8)
            display.append(
                self.plot_row_of_heatmaps(train_bbox, test_bbox, 'Compare average label bbox locations')
            )
            # bbox image values are frequency, between 0 and 100
            display[1].update_layout(coloraxis={'colorscale': 'Inferno', 'cmin': 0, 'cmax': 100},
                                     coloraxis_colorbar={'title': '% Coverage'})
            value['diff_bbox'] = self._image_diff(test_bbox, train_bbox)

        return CheckResult(value=value,
                           display=[fig.to_image('svg', width=900, height=300).decode('utf-8') for fig in display],
                           header='Heatmap Comparison')

    @staticmethod
    def plot_row_of_heatmaps(train_img: np.ndarray, test_img: np.ndarray, title: str) -> go.Figure:
        """Plot a row of heatmaps for train and test images."""
        fig = make_subplots(rows=1, cols=3, column_titles=['Train', 'Test', 'Test - Train'])
        fig.add_trace(numpy_grayscale_to_heatmap_figure(train_img), row=1, col=1)
        fig.add_trace(numpy_grayscale_to_heatmap_figure(test_img), row=1, col=2)
        fig.add_trace(numpy_grayscale_to_heatmap_figure(HeatmapComparison._image_diff(test_img, train_img)), row=1,
                      col=3)
        fig.update_yaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
        fig.update_xaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
        fig.update_layout(title=title, width=900, height=300)
        apply_heatmap_image_properties(fig)
        return fig

    @staticmethod
    def _image_diff(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Return the difference between two grayscale images as a grayscale image."""
        diff = img1.astype(np.int32) - img2.astype(np.int32)
        return np.abs(diff).astype(np.uint8)

    def _label_to_image(self, label: np.ndarray, original_shape: Tuple[int], classes_to_display: Optional[List[str]]
                        ) -> np.ndarray:
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
            if classes_to_display is not None and self._class_to_string(class_idx[i]) not in classes_to_display:
                continue
            # For each bounding box, set the pixels inside the bounding box to white
            image[y_min[i]:y_max[i], x_min[i]:x_max[i]] = 255
        return np.expand_dims(image, axis=2)

    def _label_to_image_batch(self, label_batch: List[torch.Tensor], image_batch: List[np.ndarray],
                              classes_to_display: Optional[List[str]]) -> List[np.ndarray]:
        """Convert label batch to batch of images where pixels inside the bboxes are white and the rest are black."""
        return_bbox_image_batch = []
        for image, label in zip(image_batch, label_batch):
            return_bbox_image_batch.append(
                self._label_to_image(label.detach().cpu().numpy(), image.shape[:2], classes_to_display)
            )
        return return_bbox_image_batch

    @staticmethod
    def _grayscale_sum_image(batch: Iterable[np.ndarray], target_shape: List[Tuple[int, int]] = None
                             ) -> np.ndarray:
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
                resized_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                raise NotImplementedError('Images must be RGB or grayscale')

            # reshape to one shape
            if not target_shape:
                target_shape.append(resized_img.shape[:2][::-1])
            else:
                resized_img = cv2.resize(resized_img, target_shape[0], interpolation=cv2.INTER_AREA)

            # sum images
            if summed_image is None:
                summed_image = resized_img.squeeze().astype(np.int64)
            else:
                summed_image += resized_img.squeeze().astype(np.int64)

        return summed_image
