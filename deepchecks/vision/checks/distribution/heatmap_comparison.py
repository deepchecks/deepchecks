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
from copy import copy
from typing import Dict, Hashable, Callable, Tuple, List, Union, Any, Iterable

import cv2
import torch

from deepchecks.vision.utils.image_functions import numpy_to_image_figure
from plotly.subplots import make_subplots

from deepchecks.core import DatasetKind, CheckResult
from deepchecks.core.errors import DeepchecksValueError, DeepchecksNotSupportedError
from deepchecks.vision.base import Context, TrainTestCheck
from deepchecks.utils.distribution.plot import drift_score_bar_traces
from deepchecks.utils.plot import colors
from deepchecks.vision.dataset import VisionData, TaskType
import numpy as np
from collections import Counter
import plotly.graph_objs as go

__all__ = ['HeatmapComparison']


class HeatmapComparison(TrainTestCheck):
    """
    Calculate label drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the label in test dataset, by comparing its distribution to the train
    dataset. As the label may be complex, we run different measurements on the label and check their distribution.

    A measurement on a label is any function that returns a single value or n-dimensional array of values. each value
    represents a measurement on the label, such as number of objects in image or tilt of each bounding box in image.

    There are default measurements per task:
    For classification:
    - distribution of classes

    For object detection:
    - distribution of classes
    - distribution of bounding box areas
    - distribution of number of bounding boxes per image

    For numerical distributions, we use the Earth Movers Distance.
    See https://en.wikipedia.org/wiki/Wasserstein_metric
    For categorical distributions, we use the Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf.


    Parameters
    ----------
    alternative_label_measurements : List[Dict[str, Any]], default: 10
        List of measurements. Replaces the default deepchecks measurements.
        Each measurement is dictionary with keys 'name' (str), 'method' (Callable) and is_continuous (bool),
        representing attributes of said method.
    min_sample_size: int, default: None
        number of minimum samples (not batches) to be accumulated in order to estimate the boundaries (min, max) of
        continuous histograms. As the check cannot load all label measurement results into memory, the check saves only
        the histogram of results - but prior to that, the check requires to know the estimated boundaries of train AND
        test datasets (they must share an x-axis).
    default_num_bins: int, default: 100
        number of bins to use for continuous distributions. This value is not used if the distribution has less unique
        values than default number of bins (and instead, number of unique values is used).
    """

    def __init__(self,
                 n_sample: int = 1_000,
                 random_state: int = 42):
        super().__init__()
        self.n_sample = n_sample
        self.random_state = random_state

    def initialize_run(self, context: Context):
        """Initialize run.

        Function initializes the following private variables:

        Label measurements:
        _label_measurements: all label measurements to be calculated in run
        _continuous_label_measurements: all continuous label measurements
        _discrete_label_measurements: all discrete label measurements

        Value counts of measures, to be updated per batch:
        _train_hists, _test_hists: histograms for continuous measurements for train and test respectively.
            Initialized as list of empty histograms (np.array) that update in the "update" method per batch.
        _train_counters, _test_counters: counters for discrete measurements for train and test respectively.
            Initialized as list of empty counters (collections.Counter) that update in the "update" method per batch.

        Parameters for continuous measurements' histogram calculation:
        _bounds_list: List[Tuple]. Each tuple represents histogram bounds (min, max)
        _num_bins_list: List[int]. List of number of bins for each histogram.
        _edges: List[np.array]. List of x-axis values for each histogram.
        """
        train_dataset = context.train

        self._task_type = train_dataset.task_type

        self._train_greyscale_heatmap = None
        self._test_greyscale_heatmap = None
        self._shape = []
        self._train_counter = 0
        self._test_counter = 0

        if self._task_type == TaskType.OBJECT_DETECTION:
            self._train_bbox_heatmap = None
            self._test_bbox_heatmap = None

    def update(self, context: Context, batch: Any, dataset_kind):
        """Perform update on batch for train or test counters and histograms."""

        if dataset_kind == DatasetKind.TRAIN:
            image_batch = context.train.image_formatter(batch)
            summed_image = greyscale_sum_image(image_batch, self._shape)
            if self._train_greyscale_heatmap is None:
                self._train_greyscale_heatmap = summed_image
            else:
                self._train_greyscale_heatmap += summed_image
            self._train_counter += len(image_batch)
        elif dataset_kind == DatasetKind.TEST:
            image_batch = context.test.image_formatter(batch)
            summed_image = greyscale_sum_image(image_batch, self._shape)
            if self._test_greyscale_heatmap is None:
                self._test_greyscale_heatmap = summed_image
            else:
                self._test_greyscale_heatmap += summed_image
            self._test_counter += len(image_batch)

        if self._task_type == TaskType.OBJECT_DETECTION:
            if dataset_kind == DatasetKind.TRAIN:
                label_batch = context.train.label_formatter(batch)
                label_image_batch = label_to_image_batch(label_batch, self._shape)
                summed_image = greyscale_sum_image(label_image_batch, self._shape)
                if self._train_bbox_heatmap is None:
                    self._train_bbox_heatmap = summed_image
                else:
                    self._train_bbox_heatmap += summed_image
            elif dataset_kind == DatasetKind.TEST:
                label_batch = context.test.label_formatter(batch)
                label_image_batch = label_to_image_batch(label_batch, self._shape)
                summed_image = greyscale_sum_image(label_image_batch, self._shape)
                if self._test_bbox_heatmap is None:
                    self._test_bbox_heatmap = summed_image
                else:
                    self._test_bbox_heatmap += summed_image

    def compute(self, context: Context) -> CheckResult:
        """Calculate drift for all columns.

        Returns
        -------
        CheckResult
            value: drift score.
            display: label distribution graph, comparing the train and test distributions.
        """
        train_greyscale = (np.expand_dims(self._train_greyscale_heatmap, axis=2) /
                           self._train_counter).astype(np.uint8)
        test_greyscale = (np.expand_dims(self._test_greyscale_heatmap, axis=2) /
                          self._test_counter).astype(np.uint8)
        train_bbox = (np.expand_dims(self._train_bbox_heatmap, axis=2) /
                      self._train_counter).astype(np.uint8)
        test_bbox = (np.expand_dims(self._test_bbox_heatmap, axis=2) /
                     self._test_counter).astype(np.uint8)

        if self._task_type == TaskType.OBJECT_DETECTION:
            n_rows = 2
            row_titles = ['Brightness', 'BBox Heatmap']
        else:
            n_rows = 1
            row_titles = ['Brightness']

        fig = make_subplots(rows=n_rows, cols=2, column_titles=['Train', 'Test'], row_titles=row_titles)
        fig.add_trace(numpy_to_image_figure(train_greyscale), row=1, col=1)
        fig.add_trace(numpy_to_image_figure(test_greyscale), row=1, col=2)
        if n_rows == 2:
            fig.add_trace(numpy_to_image_figure(train_bbox), row=2, col=1)
            fig.add_trace(numpy_to_image_figure(test_bbox), row=2, col=2)
        fig.update_yaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
        fig.update_xaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
        fig.update_layout(title='Comparing average images for train and test', width=600, height=300 * n_rows)

        return CheckResult(value=None, display=[fig], header='Heatmap Comparison')


def label_to_image(label: np.ndarray, shape: List[Tuple[int, int]]) -> np.ndarray:
    """Convert label array to an image where pixels inside the bboxes are white and the rest are black."""
    image = np.zeros(shape[0], dtype=np.uint8)
    label = label.reshape((-1, 5))
    x_min = (label[:, 1]).astype(np.int32)
    y_min = (label[:, 2]).astype(np.int32)
    x_max = (label[:, 1] + label[:, 3]).astype(np.int32)
    y_max = (label[:, 2] + label[:, 4]).astype(np.int32)
    for i in range(len(label)):
        image[y_min[i]:y_max[i], x_min[i]:x_max[i]] = 255
    return np.expand_dims(image, axis=2)


def label_to_image_batch(label_batch: List[torch.Tensor], shape: List[Tuple[int, int]]) -> List[np.ndarray]:
    """Convert label batch to batch of images where pixels inside the bboxes are white and the rest are black."""
    image_batch = []
    for label in label_batch:
        image_batch.append(label_to_image(label.detach().cpu().numpy(), shape))
    return image_batch


def greyscale_sum_image(batch: Iterable[np.ndarray], target_shape: List[Tuple[int, int]] = None
                        ) -> np.ndarray:
    """Sum all images in batch to one greyscale image of shape target_shape.

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

    for img in batch:
        # Cast to greyscale
        if img.shape[2] == 1:
            resized_img = img
        elif img.shape[2] == 3:
            resized_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            raise NotImplementedError('Images must be RGB or greyscale')

        # reshape to one shape
        if not target_shape:
            target_shape.append(resized_img.shape[:2][::-1])
            resized_img = resized_img
        else:
            resized_img = cv2.resize(resized_img, target_shape[0], interpolation=cv2.INTER_AREA)

        # sum images
        if summed_image is None:
            summed_image = resized_img.astype(np.int64)
        else:
            summed_image += resized_img.astype(np.int64)

    return summed_image
