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
import warnings
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List

import pandas as pd

from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.checks import CheckConfig, ReduceMixin
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepchecks.utils.distribution.drift import calc_drift_and_plot, drift_condition
from deepchecks.vision import Batch, Context, TrainTestCheck
from deepchecks.vision.utils.label_prediction_properties import (DEFAULT_CLASSIFICATION_LABEL_PROPERTIES,
                                                                 DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES,
                                                                 get_column_type, properties_flatten,
                                                                 validate_properties)
from deepchecks.vision.vision_data import TaskType

__all__ = ['TrainTestLabelDrift']


class TrainTestLabelDrift(TrainTestCheck, ReduceMixin):
    """
    Calculate label drift between train dataset and test dataset, using statistical measures.

    Check calculates a drift score for the label in test dataset, by comparing its distribution to the train
    dataset. As the label may be complex, we calculate different properties of the label and check their distribution.

    A label property is any function that gets labels and returns list of values. each
    value represents a property on the label, such as number of objects in image or tilt of each bounding box in image.

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
    label_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of:
        - 'numeric' - for continuous ordinal outputs.
        - 'categorical' - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        For more on image / label properties, see the :ref:`property guide </user-guide/vision/vision_properties.rst>`
        - 'class_id' - for properties that return the class_id. This is used because these
          properties are later matched with the VisionData.label_map, if one was given.
    margin_quantile_filter: float, default: 0.025
        float in range [0,0.5), representing which margins (high and low quantiles) of the distribution will be filtered
        out of the EMD calculation. This is done in order for extreme values not to affect the calculation
        disproportionally. This filter is applied to both distributions, in both margins.
    max_num_categories_for_drift: int, default: 10
        Only for discrete properties. Max number of allowed categories. If there are more,
        they are binned into an "Other" category. If max_num_categories=None, there is no limit. This limit applies
        for both drift calculation and for distribution plots.
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
    max_num_categories: int, default: None
        Deprecated. Please use max_num_categories_for_drift and max_num_categories_for_display instead
    """

    def __init__(
            self,
            label_properties: List[Dict[str, Any]] = None,
            margin_quantile_filter: float = 0.025,
            max_num_categories_for_drift: int = 10,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            categorical_drift_method='cramer_v',
            max_num_categories: int = None,  # Deprecated
            **kwargs
    ):
        super().__init__(**kwargs)
        # validate label properties:
        self.user_label_properties = (
            validate_properties(label_properties)
            if label_properties is not None
            else None
        )
        self.margin_quantile_filter = margin_quantile_filter
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
        self.categorical_drift_method = categorical_drift_method

        self._label_properties = None
        self._train_label_properties = None
        self._test_label_properties = None

    def initialize_run(self, context: Context):
        """Initialize run.

        Function initializes the following private variables:

        Label properties:

        _label_properties: all label properties to be calculated in run

        Label properties caching:
        _train_label_properties, _test_label_properties: Dicts of lists accumulating the label properties computed for
        each batch.
        """
        train_dataset = context.train

        task_type = train_dataset.task_type

        if self.user_label_properties is not None:
            self._label_properties = self.user_label_properties
        elif task_type == TaskType.CLASSIFICATION:
            self._label_properties = DEFAULT_CLASSIFICATION_LABEL_PROPERTIES
        elif task_type == TaskType.OBJECT_DETECTION:
            self._label_properties = DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES
        else:
            raise NotImplementedError('Check must receive either label_properties or run '
                                      'on Classification or Object Detection class')

        self._train_label_properties = defaultdict(list)
        self._test_label_properties = defaultdict(list)

    def update(self, context: Context, batch: Batch, dataset_kind):
        """Perform update on batch for train or test properties."""
        # For all transformers, calculate histograms by batch:
        if dataset_kind == DatasetKind.TRAIN:
            properties = self._train_label_properties
        elif dataset_kind == DatasetKind.TEST:
            properties = self._test_label_properties
        else:
            raise DeepchecksNotSupportedError(f'Unsupported dataset kind {dataset_kind}')

        for label_property in self._label_properties:
            # Flatten the properties since I don't care in this check about the property-per-sample coupling
            properties[label_property['name']] += properties_flatten(label_property['method'](batch.labels))

    def compute(self, context: Context) -> CheckResult:
        """Calculate drift on label properties samples that were collected during update() calls.

        Returns
        -------
        CheckResult
            value: drift score.
            display: label distribution graph, comparing the train and test distributions.
        """
        values_dict = OrderedDict()
        displays_dict = OrderedDict()
        label_properties_names = [x['name'] for x in self._label_properties]
        for label_property in self._label_properties:
            name = label_property['name']
            output_type = label_property['output_type']
            # If type is class converts to label names
            if output_type == 'class_id':
                self._train_label_properties[name] = [context.train.label_id_to_name(class_id) for class_id in
                                                      self._train_label_properties[name]]
                self._test_label_properties[name] = [context.test.label_id_to_name(class_id) for class_id in
                                                     self._test_label_properties[name]]

            value, method, display = calc_drift_and_plot(
                train_column=pd.Series(self._train_label_properties[name]),
                test_column=pd.Series(self._test_label_properties[name]),
                value_name=name,
                column_type=get_column_type(output_type),
                margin_quantile_filter=self.margin_quantile_filter,
                max_num_categories_for_drift=self.max_num_categories_for_drift,
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
            columns_order = sorted(label_properties_names, key=lambda col: values_dict[col]['Drift score'],
                                   reverse=True)

            headnote = '<span>' \
                'The Drift score is a measure for the difference between two distributions. ' \
                'In this check, drift is measured ' \
                f'for the distribution of the following label properties: {label_properties_names}.' \
                '</span>'

            displays = [headnote] + [displays_dict[col] for col in columns_order]
        else:
            displays = None

        return CheckResult(value=values_dict, display=displays, header='Train Test Label Drift')

    def config(self, include_version: bool = True) -> CheckConfig:
        if isinstance(self.user_label_properties, list):
            for prop in self.user_label_properties:
                if 'method' not in prop or not prop['method']:
                    raise ValueError('Each label property is expected to contain not emtpy "method" key')
                if callable(prop['method']):
                    type_name = type(self).__name__
                    raise ValueError(
                        f'Serialization of "{type_name}" check instance is not supported '
                        'if custom user defined properties were passed to the "label_properties" '
                        f'parameter during instance initialization. Property name: {prop["name"]}'
                    )
        return self._prepare_config(
            include_version=include_version,
            params={
                'label_properties': self.user_label_properties,
                'margin_quantile_filter': self.margin_quantile_filter,
                'max_num_categories_for_drift': self.max_num_categories_for_drift,
                'max_num_categories_for_display': self.max_num_categories_for_display,
                'show_categories_by': self.show_categories_by,
                'categorical_drift_method': self.categorical_drift_method
            },
        )

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return label drift score per label property."""
        return {name: label_property['Drift score'] for name, label_property in check_result.value.items()}

    def add_condition_drift_score_less_than(self, max_allowed_categorical_score: float = 0.15,
                                            max_allowed_numeric_score: float = 0.075,
                                            max_allowed_psi_score: float = None,
                                            max_allowed_earth_movers_score: float = None
                                            ) -> 'TrainTestLabelDrift':
        """
        Add condition - require label properties drift score to be less than a certain threshold.

        The industry standard for PSI limit is above 0.2.
        Cramer's V does not have a common industry standard.
        Earth movers does not have a common industry standard.
        The threshold was lowered by 25% compared to feature drift defaults due to the higher importance of label drift.

        Parameters
        ----------
        max_allowed_categorical_score: float , default: 0.15
            the max threshold for the PSI score
        max_allowed_numeric_score: float ,  default: 0.075
            the max threshold for the Earth Mover's Distance score
        max_allowed_psi_score: float, default None
            Deprecated. Please use max_allowed_categorical_score instead
        max_allowed_earth_movers_score: float, default None
            Deprecated. Please use max_allowed_numeric_score instead
        Returns
        -------
        ConditionResult
            False if any column has passed the max threshold, True otherwise
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

        condition = drift_condition(max_allowed_categorical_score, max_allowed_numeric_score,
                                    'label property', 'label properties')

        return self.add_condition(f'categorical drift score < {max_allowed_categorical_score} and '
                                  f'numerical drift score < {max_allowed_numeric_score}',
                                  condition)
