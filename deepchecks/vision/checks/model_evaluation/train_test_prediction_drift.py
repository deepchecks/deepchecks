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
import warnings
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List

import pandas as pd

from deepchecks import ConditionResult
from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.utils.distribution.drift import (SUPPORTED_CATEGORICAL_METHODS, SUPPORTED_NUMERIC_METHODS,
                                                 calc_drift_and_plot, get_drift_method)
from deepchecks.vision import Batch, Context, TrainTestCheck
from deepchecks.vision.utils.label_prediction_properties import (DEFAULT_CLASSIFICATION_PREDICTION_PROPERTIES,
                                                                 DEFAULT_OBJECT_DETECTION_PREDICTION_PROPERTIES,
                                                                 get_column_type, properties_flatten,
                                                                 validate_properties)
from deepchecks.vision.vision_data import TaskType

__all__ = ['TrainTestPredictionDrift']


class TrainTestPredictionDrift(TrainTestCheck):
    """
    Calculate prediction drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the predictions in the test dataset, by comparing its distribution to the
    train dataset. As the predictions may be complex, we calculate different properties of the predictions and check
    their distribution.

    A prediction property is any function that gets predictions and returns list of values. each
    value represents a property of the prediction, such as number of objects in image or tilt of each bounding box
    in image.

    There are default properties per task:
    For classification:
    - distribution of classes

    For object detection:
    - distribution of classes
    - distribution of bounding box areas
    - distribution of number of bounding boxes per image

    For numerical distributions, we use the Earth Movers Distance.
    See https://en.wikipedia.org/wiki/Wasserstein_metric

    For categorical distributions, we use the Cramer's V.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    We also support Population Stability Index (PSI).
    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf.


    Parameters
    ----------
    prediction_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of 'continuous'/'discrete'/'class_id'
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    max_num_categories_for_drift: int, default: 10
        Only for categorical columns. Max number of allowed categories. If there are more,
        they are binned into an "Other" category. If None, there is no limit.
    max_num_categories_for_display: int, default: 10
        Max number of categories to show in plot.
    show_categories_by: str, default: 'largest_difference'
        Specify which categories to show for categorical features' graphs, as the number of shown categories is limited
        by max_num_categories_for_display. Possible values:
        - 'train_largest': Show the largest train categories.
        - 'test_largest': Show the largest test categories.
        - 'largest_difference': Show the largest difference between categories.
    categorical_drift_method: str, default: "cramer_v"
        decides which method to use on categorical variables. Possible values are:
        "cramers_v" for Cramer's V, "PSI" for Population Stability Index (PSI).
    max_num_categories: int, default: None
        Deprecated. Please use max_num_categories_for_drift and max_num_categories_for_display instead
    """

    def __init__(
            self,
            prediction_properties: List[Dict[str, Any]] = None,
            margin_quantile_filter: float = 0.025,
            max_num_categories_for_drift: int = 10,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            categorical_drift_method='cramer_v',
            max_num_categories: int = None,  # Deprecated
            **kwargs
    ):
        super().__init__(**kwargs)
        # validate prediction properties:
        if prediction_properties is not None:
            validate_properties(prediction_properties)
        self.user_prediction_properties = prediction_properties
        self.margin_quantile_filter = margin_quantile_filter
        self.categorical_drift_method = categorical_drift_method
        if max_num_categories is not None:
            warnings.warn(
                f'{self.__class__.__name__}: max_num_categories is deprecated. please use max_num_categories_for_drift '
                'and max_num_categories_for_display instead',
                DeprecationWarning
            )
            max_num_categories_for_drift = max_num_categories_for_drift or max_num_categories
            max_num_categories_for_display = max_num_categories_for_display or max_num_categories
        self.max_num_categories_for_drift = max_num_categories_for_drift
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        self._prediction_properties = None
        self._train_prediction_properties = None
        self._test_prediction_properties = None

    def initialize_run(self, context: Context):
        """Initialize run.

        Function initializes the following private variables:

        Prediction properties:
        _prediction_properties: all predictions properties to be calculated in run

        Prediction properties caching: _train_prediction_properties, _test_prediction_properties: Dicts of lists
        accumulating the predictions properties computed for each batch.
        """
        train_dataset = context.train

        task_type = train_dataset.task_type

        if self.user_prediction_properties is not None:
            self._prediction_properties = self.user_prediction_properties
        elif task_type == TaskType.CLASSIFICATION:
            self._prediction_properties = DEFAULT_CLASSIFICATION_PREDICTION_PROPERTIES
        elif task_type == TaskType.OBJECT_DETECTION:
            self._prediction_properties = DEFAULT_OBJECT_DETECTION_PREDICTION_PROPERTIES
        else:
            raise NotImplementedError('Check must receive either prediction_properties or '
                                      'run on Classification or Object Detection class')

        self._train_prediction_properties = defaultdict(list)
        self._test_prediction_properties = defaultdict(list)

    def update(self, context: Context, batch: Batch, dataset_kind):
        """Perform update on batch for train or test properties."""
        # For all transformers, calculate histograms by batch:
        if dataset_kind == DatasetKind.TRAIN:
            properties = self._train_prediction_properties
        elif dataset_kind == DatasetKind.TEST:
            properties = self._test_prediction_properties
        else:
            raise DeepchecksNotSupportedError(f'Unsupported dataset kind {dataset_kind}')

        for prediction_property in self._prediction_properties:
            # Flatten the properties since I don't care in this check about the property-per-sample coupling
            properties[prediction_property['name']] += properties_flatten(
                prediction_property['method'](batch.predictions)
            )

    def compute(self, context: Context) -> CheckResult:
        """Calculate drift on prediction properties samples that were collected during update() calls.

        Returns
        -------
        CheckResult
            value: drift score.
            display: label distribution graph, comparing the train and test distributions.
        """
        values_dict = OrderedDict()
        displays_dict = OrderedDict()
        prediction_properties_names = [x['name'] for x in self._prediction_properties]
        for prediction_property in self._prediction_properties:
            name = prediction_property['name']
            output_type = prediction_property['output_type']
            # If type is class converts to label names
            if output_type == 'class_id':
                self._train_prediction_properties[name] = [context.train.label_id_to_name(class_id) for class_id in
                                                           self._train_prediction_properties[name]]
                self._test_prediction_properties[name] = [context.test.label_id_to_name(class_id) for class_id in
                                                          self._test_prediction_properties[name]]

            value, method, display = calc_drift_and_plot(
                train_column=pd.Series(self._train_prediction_properties[name]),
                test_column=pd.Series(self._test_prediction_properties[name]),
                value_name=name,
                column_type=get_column_type(output_type),
                margin_quantile_filter=self.margin_quantile_filter,
                max_num_categories_for_drift=self.max_num_categories_for_drift,
                max_num_categories_for_display=self.max_num_categories_for_display,
                show_categories_by=self.show_categories_by,
                categorical_drift_method=self.categorical_drift_method,
            )
            values_dict[name] = {
                'Drift score': value,
                'Method': method,
            }
            displays_dict[name] = display

        columns_order = sorted(prediction_properties_names, key=lambda col: values_dict[col]['Drift score'],
                               reverse=True)

        headnote = '<span>' \
                   'The Drift score is a measure for the difference between two distributions. ' \
                   'In this check, drift is measured ' \
                   f'for the distribution of the following prediction properties: {prediction_properties_names}.' \
                   '</span>'

        displays = [headnote] + [displays_dict[col] for col in columns_order]

        return CheckResult(value=values_dict, display=displays, header='Train Test Prediction Drift')

    def add_condition_drift_score_not_greater_than(self, max_allowed_categorical_score: float = 0.15,
                                                   max_allowed_numeric_score: float = 0.075,
                                                   max_allowed_psi_score: float = None,
                                                   max_allowed_earth_movers_score: float = None
                                                   ) -> 'TrainTestPredictionDrift':
        """
        Add condition - require prediction properties drift score to not be more than a certain threshold.

        The industry standard for PSI limit is above 0.2.
        Cramer's V does not have a common industry standard.
        Earth movers does not have a common industry standard.
        The threshold was lowered by 25% compared to feature drift defaults due to the higher importance of prediction
        drift.

        Parameters
        ----------
        max_allowed_categorical_score: float , default: 0.15
            the max threshold for the categorical variable drift score
        max_allowed_numeric_score: float ,  default: 0.075
            the max threshold for the numeric variable drift score
        max_allowed_psi_score: float, default None
            Deprecated. Please use max_allowed_categorical_score instead
        max_allowed_earth_movers_score: float, default None
            Deprecated. Please use max_allowed_numeric_score instead
        Returns
        -------
        ConditionResult
            False if any property has passed the max threshold, True otherwise
        """
        if max_allowed_psi_score is not None:
            warnings.warn(
                f'{self.__class__.__name__}: max_allowed_psi_score is deprecated. please use '
                f'max_allowed_categorical_score instead',
                DeprecationWarning
            )
            if max_allowed_categorical_score is not None:
                max_allowed_categorical_score = max_allowed_psi_score
        if max_allowed_earth_movers_score is not None:
            warnings.warn(
                f'{self.__class__.__name__}: max_allowed_earth_movers_score is deprecated. please use '
                f'max_allowed_numeric_score instead',
                DeprecationWarning
            )
            if max_allowed_numeric_score is not None:
                max_allowed_numeric_score = max_allowed_earth_movers_score

        def condition(result: Dict) -> ConditionResult:
            cat_method, num_method = get_drift_method(result)
            not_passing_categorical_columns = {props: f'{d["Drift score"]:.2}' for props, d in result.items() if
                                               d['Drift score'] > max_allowed_categorical_score
                                               and d['Method'] in SUPPORTED_CATEGORICAL_METHODS}
            not_passing_numeric_columns = {props: f'{d["Drift score"]:.2}' for props, d in result.items() if
                                           d['Drift score'] > max_allowed_numeric_score
                                           and d['Method'] in SUPPORTED_NUMERIC_METHODS}
            return_str = ''
            if not_passing_categorical_columns:
                return_str += f'Found categorical prediction properties with {cat_method} above threshold:' \
                              f' {not_passing_categorical_columns}\n'
            if not_passing_numeric_columns:
                return_str += f'Found numeric prediction properties with {num_method} above' \
                              f' threshold: {not_passing_numeric_columns}\n'

            if return_str:
                return ConditionResult(ConditionCategory.FAIL, return_str)
            else:
                return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(f'categorical drift score <= {max_allowed_categorical_score} and '
                                  f'numerical drift score <= {max_allowed_numeric_score} for prediction drift',
                                  condition)
