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
from typing import Dict, Hashable, Callable, Tuple, List, Union, Any

import pandas as pd
from deepchecks.utils.distribution.drift import calc_drift_and_plot
from plotly.subplots import make_subplots

from deepchecks.core import DatasetKind, CheckResult
from deepchecks.core.errors import DeepchecksValueError, DeepchecksNotSupportedError
from deepchecks.vision.base import Context, TrainTestCheck
from deepchecks.utils.distribution.plot import drift_score_bar_traces
from deepchecks.utils.plot import colors
from deepchecks.vision.dataset import VisionData, TaskType
import numpy as np
from collections import Counter, OrderedDict
import plotly.graph_objs as go

__all__ = ['TrainTestLabelDrift']


# TODO: Add label sampling when available

# Functions temporarily here, will be changed when Label and Prediction classes exist:
def _get_bbox_area(label, _):
    """Return a list containing the area of bboxes per image in batch."""
    areas = (label.reshape((-1, 5))[:, 4] * label.reshape((-1, 5))[:, 3]).reshape(-1, 1).tolist()
    return areas


def _count_num_bboxes(label, _):
    """Return a list containing the number of bboxes per image in batch."""
    num_bboxes = label.shape[0]
    return num_bboxes


def _get_samples_per_class_classification(label, dataset):
    """Return a list containing the class per image in batch."""
    return dataset.label_id_to_name(label.tolist())


def _get_samples_per_class_object_detection(label, dataset):
    """Return a list containing the class per image in batch."""
    return [[dataset.label_id_to_name(arr.reshape((-1, 5))[:, 0])] for arr in label]


DEFAULT_CLASSIFICATION_LABEL_MEASUREMENTS = [
    {'name': 'Samples per class', 'method': _get_samples_per_class_classification, 'is_continuous': False}
]

DEFAULT_OBJECT_DETECTION_LABEL_MEASUREMENTS = [
    {'name': 'Samples per class', 'method': _get_samples_per_class_object_detection, 'is_continuous': False},
    {'name': 'Bounding box area (in pixels)', 'method': _get_bbox_area, 'is_continuous': True},
    {'name': 'Number of bounding boxes per image', 'method': _count_num_bboxes, 'is_continuous': True},
]


class TrainTestLabelDrift(TrainTestCheck):
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

    def __init__(
            self,
            alternative_label_measurements: List[Dict[str, Any]] = None,
            max_num_categories: int = 10
    ):
        super().__init__()
        # validate alternative_label_measurements:
        if alternative_label_measurements is not None:
            self._validate_label_measurements(alternative_label_measurements)
        self.alternative_label_measurements = alternative_label_measurements
        self.max_num_categories = max_num_categories

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

        Parameters for continuous measurements histogram calculation:
        _bounds_list: List[Tuple]. Each tuple represents histogram bounds (min, max)
        _num_bins_list: List[int]. List of number of bins for each histogram.
        _edges: List[np.array]. List of x-axis values for each histogram.
        """
        train_dataset = context.train

        task_type = train_dataset.task_type

        if self.alternative_label_measurements is not None:
            self._label_measurements = self.alternative_label_measurements
        elif task_type == TaskType.CLASSIFICATION:
            self._label_measurements = DEFAULT_CLASSIFICATION_LABEL_MEASUREMENTS
        elif task_type == TaskType.OBJECT_DETECTION:
            self._label_measurements = DEFAULT_OBJECT_DETECTION_LABEL_MEASUREMENTS
        else:
            raise NotImplementedError('TrainTestLabelDrift must receive either alternative_label_measurements or run '
                                      'on Classification or Object Detection class')

        self._train_label_properties = OrderedDict([(k['name'], []) for k in self._label_measurements])
        self._test_label_properties = OrderedDict([(k['name'], []) for k in self._label_measurements])

    def update(self, context: Context, batch: Any, dataset_kind):
        """Perform update on batch for train or test counters and histograms."""
        # For all transformers, calculate histograms by batch:
        if dataset_kind == DatasetKind.TRAIN:
            dataset = context.train
            properties = self._train_label_properties
        elif dataset_kind == DatasetKind.TEST:
            dataset = context.test
            properties = self._test_label_properties
        else:
            raise DeepchecksNotSupportedError(f'Unsupported dataset kind {dataset_kind}')

        for label_measurement in self._label_measurements:
            properties[label_measurement['name']] += get_results_on_batch(batch, label_measurement['method'], dataset)

    def compute(self, context: Context) -> CheckResult:
        """Calculate drift on label measurements histograms that were collected during update() calls.

        Returns
        -------
        CheckResult
            value: drift score.
            display: label distribution graph, comparing the train and test distributions.
        """

        values_dict = OrderedDict()
        displays_dict = OrderedDict()
        label_measures_names = [x['name'] for x in self._label_measurements]
        for label_measure in self._label_measurements:
            measure_name = label_measure['name']

            value, method, display = calc_drift_and_plot(
                train_column=pd.Series(self._train_label_properties[measure_name]),
                test_column=pd.Series(self._test_label_properties[measure_name]),
                plot_title=measure_name,
                column_type='numerical' if label_measure['is_continuous'] else 'categorical',
                max_num_categories=self.max_num_categories
            )
            values_dict[measure_name] = {
                'Drift score': value,
                'Method': method,
            }
            displays_dict[measure_name] = display

        columns_order = sorted(label_measures_names, key=lambda col: values_dict[col]['Drift score'], reverse=True)

        headnote = '<span>' \
                   'The Drift score is a measure for the difference between two distributions. ' \
                   'In this check, drift is measured ' \
                   f'for the distribution of the following label properties: {label_measures_names}.' \
                   '</span>'

        displays = [headnote] + [displays_dict[col] for col in columns_order]

        return CheckResult(value=values_dict, display=displays, header='Train Test Label Drift')

    @staticmethod
    def _validate_label_measurements(label_measurements):
        """Validate structure of label measurements."""
        expected_keys = ['name', 'method', 'is_continuous']
        if not isinstance(label_measurements, list):
            raise DeepchecksValueError(
                f'Expected label measurements to be a list, instead got {label_measurements.__class__.__name__}')
        for label_measurement in label_measurements:
            if not isinstance(label_measurement, dict) or any(
                    key not in label_measurement.keys() for key in expected_keys):
                raise DeepchecksValueError(f'Label measurement must be of type dict, and include keys {expected_keys}')


def get_results_on_batch(batch, label_measurement, dataset: VisionData):
    """Calculate transformer result on batch of labels."""
    calc_res = [label_measurement(arr, dataset) for arr in dataset.label_formatter(batch)]
    if len(calc_res) != 0 and isinstance(calc_res[0], list):
        calc_res = [x[0] for x in sum(calc_res, [])]
    return calc_res
