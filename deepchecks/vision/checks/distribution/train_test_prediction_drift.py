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
"""Module contains Train Test Prediction Drift check."""
from typing import Dict, List, Any

import pandas as pd

from deepchecks import ConditionResult
from deepchecks.utils.distribution.drift import calc_drift_and_plot

from deepchecks.core import DatasetKind, CheckResult
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.vision.base import Context, TrainTestCheck
from deepchecks.vision.dataset import TaskType
from deepchecks.vision.utils.measurements import validate_measurements, \
    DEFAULT_CLASSIFICATION_PREDICTION_MEASUREMENTS, DEFAULT_OBJECT_DETECTION_PREDICTION_MEASUREMENTS, \
    get_prediction_measurements_on_batch
from collections import OrderedDict


__all__ = ['TrainTestPredictionDrift']


class TrainTestPredictionDrift(TrainTestCheck):
    """
    Calculate prediction drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the predictions in the test dataset, by comparing its distribution to the
    train dataset. As the predictions may be complex, we run different measurements on the predictions and check
    their distribution.

    A measurement on a prediction is any function that returns a single value or n-dimensional array of values. each
    value represents a measurement on the prediction, such as number of objects in image or tilt of each bounding box
    in image.

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
    alternative_prediction_measurements : List[Dict[str, Any]], default: None
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
            alternative_prediction_measurements: List[Dict[str, Any]] = None,
            max_num_categories: int = 10
    ):
        super().__init__()
        # validate alternative_prediction_measurements:
        if alternative_prediction_measurements is not None:
            validate_measurements(alternative_prediction_measurements)
        self.alternative_prediction_measurements = alternative_prediction_measurements
        self.max_num_categories = max_num_categories

    def initialize_run(self, context: Context):
        """Initialize run.

        Function initializes the following private variables:

        Prediction measurements:
        _prediction_measurements: all label measurements to be calculated in run

        Prediction measurements caching: _train_prediction_properties, _test_prediction_properties: Dicts of lists
        accumulating the label measurements computed for each batch.
        """
        train_dataset = context.train

        task_type = train_dataset.task_type

        if self.alternative_prediction_measurements is not None:
            self._prediction_measurements = self.alternative_prediction_measurements
        elif task_type == TaskType.CLASSIFICATION:
            self._prediction_measurements = DEFAULT_CLASSIFICATION_PREDICTION_MEASUREMENTS
        elif task_type == TaskType.OBJECT_DETECTION:
            self._prediction_measurements = DEFAULT_OBJECT_DETECTION_PREDICTION_MEASUREMENTS
        else:
            raise NotImplementedError('TrainTestLabelDrift must receive either alternative_prediction_measurements or '
                                      'run on Classification or Object Detection class')

        self._train_prediction_properties = OrderedDict([(k['name'], []) for k in self._prediction_measurements])
        self._test_prediction_properties = OrderedDict([(k['name'], []) for k in self._prediction_measurements])

    def update(self, context: Context, batch: Any, dataset_kind):
        """Perform update on batch for train or test properties."""
        # For all transformers, calculate histograms by batch:
        if dataset_kind == DatasetKind.TRAIN:
            dataset = context.train
            properties = self._train_prediction_properties
        elif dataset_kind == DatasetKind.TEST:
            dataset = context.test
            properties = self._test_prediction_properties
        else:
            raise DeepchecksNotSupportedError(f'Unsupported dataset kind {dataset_kind}')

        for prediction_measurement in self._prediction_measurements:
            properties[prediction_measurement['name']].extend(
                get_prediction_measurements_on_batch(batch, prediction_measurement['method'], dataset, context)
            )

    def compute(self, context: Context) -> CheckResult:
        """Calculate drift on prediction measurements samples that were collected during update() calls.

        Returns
        -------
        CheckResult
            value: drift score.
            display: label distribution graph, comparing the train and test distributions.
        """
        values_dict = OrderedDict()
        displays_dict = OrderedDict()
        prediction_measures_names = [x['name'] for x in self._prediction_measurements]
        for prediction_measure in self._prediction_measurements:
            measure_name = prediction_measure['name']

            value, method, display = calc_drift_and_plot(
                train_column=pd.Series(self._train_prediction_properties[measure_name]),
                test_column=pd.Series(self._test_prediction_properties[measure_name]),
                plot_title=measure_name,
                column_type='numerical' if prediction_measure['is_continuous'] else 'categorical',
                max_num_categories=self.max_num_categories
            )
            values_dict[measure_name] = {
                'Drift score': value,
                'Method': method,
            }
            displays_dict[measure_name] = display

        columns_order = sorted(prediction_measures_names, key=lambda col: values_dict[col]['Drift score'], reverse=True)

        headnote = '<span>' \
                   'The Drift score is a measure for the difference between two distributions. ' \
                   'In this check, drift is measured ' \
                   f'for the distribution of the following prediction properties: {prediction_measures_names}.' \
                   '</span>'

        displays = [headnote] + [displays_dict[col] for col in columns_order]

        return CheckResult(value=values_dict, display=displays, header='Train Test Prediction Drift')

    def add_condition_drift_score_not_greater_than(self, max_allowed_psi_score: float = 0.15,
                                                   max_allowed_earth_movers_score: float = 0.075
                                                   ) -> 'TrainTestPredictionDrift':
        """
        Add condition - require prediction measurements drift score to not be more than a certain threshold.

        The industry standard for PSI limit is above 0.2.
        Earth movers does not have a common industry standard.
        The threshold was lowered by 25% compared to feature drift defaults due to the higher importance of prediction
        drift.

        Parameters
        ----------
        max_allowed_psi_score: float , default: 0.15
            the max threshold for the PSI score
        max_allowed_earth_movers_score: float , default: 0.075
            the max threshold for the Earth Mover's Distance score
        Returns
        -------
        ConditionResult
            False if any measurement has passed the max threshold, True otherwise
        """

        def condition(result: Dict) -> ConditionResult:
            not_passing_categorical_columns = {measures: f'{d["Drift score"]:.2}' for measures, d in result.items() if
                                               d['Drift score'] > max_allowed_psi_score and d['Method'] == 'PSI'}
            not_passing_numeric_columns = {measures: f'{d["Drift score"]:.2}' for measures, d in result.items() if
                                           d['Drift score'] > max_allowed_earth_movers_score
                                           and d['Method'] == "Earth Mover's Distance"}
            return_str = ''
            if not_passing_categorical_columns:
                return_str += f'Found non-continues prediction measurements with PSI drift score above threshold:' \
                              f' {not_passing_categorical_columns}\n'
            if not_passing_numeric_columns:
                return_str += f'Found continues prediction measurements with Earth Mover\'s drift score above' \
                              f' threshold: {not_passing_numeric_columns}\n'

            if return_str:
                return ConditionResult(False, return_str)
            else:
                return ConditionResult(True)

        return self.add_condition(f'PSI <= {max_allowed_psi_score} and Earth Mover\'s Distance <= '
                                  f'{max_allowed_earth_movers_score} for prediction drift',
                                  condition)
