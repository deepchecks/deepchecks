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
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.checks import CheckConfig
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.core.reduce_classes import ReducePropertyMixin
from deepchecks.utils.distribution.drift import calc_drift_and_plot, drift_condition, get_drift_plot_sidenote
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.base_checks import TrainTestCheck
from deepchecks.vision.context import Context
from deepchecks.vision.utils.label_prediction_properties import (DEFAULT_CLASSIFICATION_PREDICTION_PROPERTIES,
                                                                 DEFAULT_OBJECT_DETECTION_PREDICTION_PROPERTIES,
                                                                 DEFAULT_SEMANTIC_SEGMENTATION_PREDICTION_PROPERTIES,
                                                                 get_column_type, properties_flatten)
from deepchecks.vision.utils.vision_properties import PropertiesInputType
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper

__all__ = ['TrainTestPredictionDrift']


@docstrings
class TrainTestPredictionDrift(TrainTestCheck, ReducePropertyMixin):
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

    For categorical prediction properties, it is recommended to use Cramer's V, unless your variable includes categories
    with a small number of samples (common practice is categories with less than 5 samples).
    However, in cases of a variable with many categories with few samples, it is still recommended to use Cramer's V.


    Parameters
    ----------
    prediction_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        - ``'class_id'`` - for properties that return the class_id. This is used because these
          properties are later matched with the ``VisionData.label_map``, if one was given.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    min_category_size_ratio: float, default 0.01
        minimum size ratio for categories. Categories with size ratio lower than this number are binned
        into an "Other" category.
    max_num_categories_for_drift: int, default: None
        Only for discrete properties. Max number of allowed categories. If there are more,
        they are binned into an "Other" category. This limit applies for both drift calculation and distribution plots.
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
        "cramer_v" for Cramer's V, "PSI" for Population Stability Index (PSI).
    aggregation_method: str, default: 'none'
        {property_aggregation_method_argument:2*indent}
    {additional_check_init_params:2*indent}
    """

    def __init__(
            self,
            prediction_properties: List[Dict[str, Any]] = None,
            margin_quantile_filter: float = 0.025,
            max_num_categories_for_drift: int = None,
            min_category_size_ratio: float = 0.01,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            categorical_drift_method: str = 'cramer_v',
            aggregation_method: str = 'none',
            n_samples: Optional[int] = 10000,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.prediction_properties = prediction_properties
        self.margin_quantile_filter = margin_quantile_filter
        self.categorical_drift_method = categorical_drift_method
        self.max_num_categories_for_drift = max_num_categories_for_drift
        self.min_category_size_ratio = min_category_size_ratio
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        self.aggregation_method = aggregation_method

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

        if self.prediction_properties is None:
            if task_type == TaskType.CLASSIFICATION:
                self.prediction_properties = DEFAULT_CLASSIFICATION_PREDICTION_PROPERTIES
            elif task_type == TaskType.OBJECT_DETECTION:
                self.prediction_properties = DEFAULT_OBJECT_DETECTION_PREDICTION_PROPERTIES
            elif task_type == TaskType.SEMANTIC_SEGMENTATION:
                self.prediction_properties = DEFAULT_SEMANTIC_SEGMENTATION_PREDICTION_PROPERTIES
            else:
                raise NotImplementedError('Check must receive either prediction_properties or '
                                          'run on Classification or Object Detection class')

        self._train_prediction_properties = defaultdict(list)
        self._test_prediction_properties = defaultdict(list)

    def update(self, context: Context, batch: BatchWrapper, dataset_kind):
        """Perform update on batch for train or test properties."""
        # For all transformers, calculate histograms by batch:
        if dataset_kind == DatasetKind.TRAIN:
            properties_results = self._train_prediction_properties
        elif dataset_kind == DatasetKind.TEST:
            properties_results = self._test_prediction_properties
        else:
            raise DeepchecksNotSupportedError(f'Unsupported dataset kind {dataset_kind}')

        batch_properties = batch.vision_properties(self.prediction_properties, PropertiesInputType.PREDICTIONS)

        for prop_name, prop_value in batch_properties.items():
            # Flatten the properties since we don't care in this check about the property-per-sample coupling
            properties_results[prop_name] += properties_flatten(prop_value)

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
        prediction_properties_names = [x['name'] for x in self.prediction_properties]
        for prediction_property in self.prediction_properties:
            name = prediction_property['name']
            output_type = prediction_property['output_type']
            # If type is class converts to label names
            if output_type == 'class_id':
                self._train_prediction_properties[name] = [context.train.label_map[class_id] for class_id in
                                                           self._train_prediction_properties[name]]
                self._test_prediction_properties[name] = [context.test.label_map[class_id] for class_id in
                                                          self._test_prediction_properties[name]]

            value, method, display = calc_drift_and_plot(
                train_column=pd.Series(self._train_prediction_properties[name]),
                test_column=pd.Series(self._test_prediction_properties[name]),
                value_name=name,
                column_type=get_column_type(output_type),
                margin_quantile_filter=self.margin_quantile_filter,
                max_num_categories_for_drift=self.max_num_categories_for_drift,
                min_category_size_ratio=self.min_category_size_ratio,
                max_num_categories_for_display=self.max_num_categories_for_display,
                show_categories_by=self.show_categories_by,
                categorical_drift_method=self.categorical_drift_method,
                with_display=context.with_display,
            )
            values_dict[name] = {
                'Drift score': value,
                'Method': method,
            }
            displays_dict[name] = display

        if context.with_display:
            columns_order = sorted(prediction_properties_names, key=lambda col: values_dict[col]['Drift score'],
                                   reverse=True)

            headnote = ['<span>'
                        'The Drift score is a measure for the difference between two distributions. '
                        'In this check, drift is measured for the distribution of the following '
                        f'prediction properties: {prediction_properties_names}. </span>',
                        get_drift_plot_sidenote(self.max_num_categories_for_display, self.show_categories_by)]

            displays = headnote + [displays_dict[col] for col in columns_order]
        else:
            displays = None

        return CheckResult(value=values_dict, display=displays, header='Train Test Prediction Drift')

    def config(self, include_version: bool = True, include_defaults: bool = True) -> CheckConfig:
        """Return check configuration."""
        # NOTE: prediction_properties if passed always contain callables
        if self.prediction_properties is not None:
            raise ValueError(
                'Serialization of check instances with provided '
                '"prediction_properties" parameter is not supported'
            )
        return super().config(include_version=include_version, include_defaults=include_defaults)

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return prediction drift score per prediction property."""
        value_per_property = {name: predict_property['Drift score'] for name, predict_property in
                              check_result.value.items()}
        return self.property_reduce(self.aggregation_method, pd.Series(value_per_property), 'Drift Score')

    def add_condition_drift_score_less_than(self, max_allowed_categorical_score: float = 0.15,
                                            max_allowed_numeric_score: float = 0.075) -> 'TrainTestPredictionDrift':
        """
        Add condition - require prediction properties drift score to be less than the threshold.

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

        Returns
        -------
        ConditionResult
            False if any property has passed the max threshold, True otherwise
        """
        condition = drift_condition(max_allowed_categorical_score, max_allowed_numeric_score,
                                    'prediction property', 'prediction properties')

        return self.add_condition(f'categorical drift score < {max_allowed_categorical_score} and '
                                  f'numerical drift score < {max_allowed_numeric_score} for prediction drift',
                                  condition)
