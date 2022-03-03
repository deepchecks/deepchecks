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
from typing import Dict, List, Any

import pandas as pd
from deepchecks.utils.distribution.drift import calc_drift_and_plot

from deepchecks.core import DatasetKind, CheckResult
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.vision.base import Context, TrainTestCheck
from deepchecks.vision.dataset import TaskType
from collections import OrderedDict

__all__ = ['TrainTestLabelDrift']

from deepchecks.vision.utils.measurements import DEFAULT_CLASSIFICATION_LABEL_MEASUREMENTS, \
    DEFAULT_OBJECT_DETECTION_LABEL_MEASUREMENTS, get_label_measurements_on_batch, validate_measurements


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
    max_num_categories : int , default: 10
        Only for non-continues measurements. Max number of allowed categories. If there are more,
        they are binned into an "Other" category. If max_num_categories=None, there is no limit. This limit applies
        for both drift calculation and for distribution plots.
    """

    def __init__(
            self,
            alternative_label_measurements: List[Dict[str, Any]] = None,
            max_num_categories: int = 10
    ):
        super().__init__()
        # validate alternative_label_measurements:
        if alternative_label_measurements is not None:
            validate_measurements(alternative_label_measurements)
        self.alternative_label_measurements = alternative_label_measurements
        self.max_num_categories = max_num_categories

    def initialize_run(self, context: Context):
        """Initialize run.

        Function initializes the following private variables:

        Label measurements:
        _label_measurements: all label measurements to be calculated in run

        Label measurements caching:
        _train_label_properties, _test_label_properties: Dicts of lists accumulating the label measurements computed for
        each batch.
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
        """Perform update on batch for train or test properties."""
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
            properties[label_measurement['name']].extend(
                get_label_measurements_on_batch(batch, label_measurement['method'], dataset)
            )

    def compute(self, context: Context) -> CheckResult:
        """Calculate drift on label measurement samples that were collected during update() calls.

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
