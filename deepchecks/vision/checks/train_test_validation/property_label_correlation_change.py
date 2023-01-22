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
"""Module contains the property label correlation change check."""
from collections import defaultdict
from typing import Any, Dict, Hashable, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.check_utils.feature_label_correlation_utils import (get_feature_label_correlation,
                                                                         get_feature_label_correlation_per_class)
from deepchecks.core.condition import ConditionCategory
from deepchecks.utils.dict_funcs import get_dict_entry_by_value
from deepchecks.utils.strings import format_number
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.base_checks import TrainTestCheck
from deepchecks.vision.context import Context
from deepchecks.vision.utils.property_label_correlation_utils import calc_properties_for_property_label_correlation
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper

__all__ = ['PropertyLabelCorrelationChange']

pps_url = 'https://docs.deepchecks.com/en/stable/checks_gallery/vision/' \
          'train_test_validation/plot_feature_label_correlation_change.html'
pps_html = f'<a href={pps_url} target="_blank">Predictive Power Score</a>'

PLC = TypeVar('PLC', bound='PropertyLabelCorrelationChange')


@docstrings
class PropertyLabelCorrelationChange(TrainTestCheck):
    """
    Return the Predictive Power Score of image properties, in order to estimate their ability to predict the label.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    In this check, we specifically use it to assess the ability to predict the label by an image property (e.g.
    brightness, contrast etc.)
    A high PPS (close to 1) can mean that there's a bias in the dataset, as a single property can predict the label
    successfully, using simple classic ML algorithms - meaning that a deep learning algorithm may accidentally learn
    these properties instead of more accurate complex abstractions.
    For example, in a classification dataset of wolves and dogs photographs, if only wolves are photographed in the
    snow, the brightness of the image may be used to predict the label "wolf" easily. In this case, a model might not
    learn to discern wolf from dog by the animal's characteristics, but by using the background color.

    When we compare train PPS to test PPS, A high difference can strongly indicate bias in the datasets,
    as a property that was "powerful" in train but not in test can be explained by bias in train that does
    not affect a new dataset.

    For classification tasks, this check uses PPS to predict the class by image properties.
    For object detection tasks, this check uses PPS to predict the class of each bounding box, by the image properties
    of that specific bounding box.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Parameters
    ----------
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
    per_class : bool, default: True
        boolean that indicates whether the results of this check should be calculated for all classes or per class in
        label. If True, the conditions will be run per class as well.
    n_top_properties: int, default: 5
        Number of features to show, sorted by the magnitude of difference in PPS
    random_state: int, default: None
        Random state for the ppscore.predictors function
    min_pps_to_show: float, default 0.05
            Minimum PPS to show a class in the graph
    ppscore_params: dict, default: None
        dictionary of additional parameters for the ppscore predictor function
    {additional_check_init_params:2*indent}
    """

    def __init__(
            self,
            image_properties: Optional[List[Dict[str, Any]]] = None,
            n_top_properties: int = 3,
            per_class: bool = True,
            min_pps_to_show: float = 0.05,
            ppscore_params: dict = None,
            n_samples: Optional[int] = 10000,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.image_properties = image_properties
        self.min_pps_to_show = min_pps_to_show
        self.per_class = per_class
        self.n_top_properties = n_top_properties
        self.ppscore_params = ppscore_params or {}
        self._train_properties, self._test_properties = defaultdict(list), defaultdict(list)
        self._train_properties['target'], self._test_properties['target'] = [], []

    def initialize_run(self, context: Context):
        """Initialize run."""
        context.assert_task_type(TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION)

    def update(self, context: Context, batch: BatchWrapper, dataset_kind: DatasetKind):
        """Calculate image properties for train or test batches."""
        if dataset_kind == DatasetKind.TRAIN:
            properties_results = self._train_properties
        else:
            properties_results = self._test_properties

        vision_data = context.get_data_by_kind(dataset_kind)
        data_for_properties, target = calc_properties_for_property_label_correlation(
            vision_data.task_type, batch, self.image_properties)
        properties_results['target'] += target

        for prop_name, property_values in data_for_properties.items():
            properties_results[prop_name].extend(property_values)

    def compute(self, context: Context) -> CheckResult:
        """Calculate the PPS between each property and the label.

        Returns
        -------
        CheckResult
            value: dictionaries of PPS values for train, test and train-test difference.
            display: bar graph of the PPS of each feature.
        """
        df_train = pd.DataFrame(self._train_properties)
        df_test = pd.DataFrame(self._test_properties)
        dataset_names = (context.train.name, context.test.name)
        # PPS task type is inferred from label dtype. For supported task types (object detection, classification),
        # the label should be regarded as categorical thus it is cast to object dtype.
        df_train['target'] = df_train['target'].apply(context.train.label_map.get).astype('object')
        df_test['target'] = df_test['target'].apply(context.test.label_map.get).astype('object')

        text = [
            'The Predictive Power Score (PPS) is used to estimate the ability of an image property (such as brightness)'
            f'to predict the label by itself. (Read more about {pps_html})'
            '',
            '<u>In the graph above</u>, we should suspect we have problems in our data if:',
            ''
            '1. <b>Train dataset PPS values are high</b>:',
            '   A high PPS (close to 1) can mean that there\'s a bias in the dataset, as a single property can predict'
            '   the label successfully, using simple classic ML algorithms',
            '2. <b>Large difference between train and test PPS</b> (train PPS is larger):',
            '   An even more powerful indication of dataset bias, as an image property that was powerful in train',
            '   but not in test can be explained by bias in train that is not relevant to a new dataset.',
            '3. <b>Large difference between test and train PPS</b> (test PPS is larger):',
            '   An anomalous value, could indicate drift in test dataset that caused a coincidental correlation to '
            'the target label.'
        ]

        if self.per_class is True:
            ret_value, display = get_feature_label_correlation_per_class(df_train,
                                                                         'target',
                                                                         df_test,
                                                                         'target',
                                                                         self.ppscore_params,
                                                                         self.n_top_properties,
                                                                         min_pps_to_show=self.min_pps_to_show,
                                                                         random_state=context.random_state,
                                                                         with_display=context.with_display,
                                                                         dataset_names=dataset_names)

        else:
            ret_value, display = get_feature_label_correlation(df_train,
                                                               'target',
                                                               df_test,
                                                               'target',
                                                               self.ppscore_params,
                                                               self.n_top_properties,
                                                               min_pps_to_show=self.min_pps_to_show,
                                                               random_state=context.random_state,
                                                               with_display=context.with_display,
                                                               dataset_names=dataset_names)

        if display:
            display += text

        return CheckResult(value=ret_value, display=display, header='Property Label Correlation Change')

    def add_condition_property_pps_difference_less_than(self: PLC, threshold: float = 0.2,
                                                        include_negative_diff: bool = False) -> PLC:
        """Add new condition.

        Add condition that will check that difference between train
        dataset property pps and test dataset property pps is less than X. If per_class is True, the condition
        will apply per class, and a single class with pps difference greater than X will be enough to fail the
        condition.

        Parameters
        ----------
        threshold : float , default: 0.2
            train test ps difference upper bound.
        include_negative_diff: bool, default True
            This parameter decides whether the condition checks the absolute value of the difference, or just the
            positive value.
            The difference is calculated as train PPS minus test PPS. This is because we're interested in the case
            where the test dataset is less predictive of the label than the train dataset, as this could indicate
            leakage of labels into the train dataset.


        Returns
        -------
        SFC
        """

        def condition(value: Union[Dict[Hashable, Dict[Hashable, float]],
                                   Dict[Hashable, Dict[Hashable, Dict[Hashable, float]]]],
                      ) -> ConditionResult:

            def above_threshold_fn(pps):
                return np.abs(pps) >= threshold if include_negative_diff else pps >= threshold

            if self.per_class:
                failed_features = {}
                overall_max_pps = -np.inf, ''
                for feature, pps_info in value.items():
                    per_class_diff: dict = pps_info['train-test difference']
                    failed_classes = {class_name: format_number(v) for class_name, v in per_class_diff.items()
                                      if above_threshold_fn(v)}
                    if failed_classes:
                        failed_features[feature] = failed_classes
                    # Get max diff to display when condition is passing
                    max_class, max_pps = get_dict_entry_by_value(per_class_diff)
                    if max_pps > overall_max_pps[0]:
                        overall_max_pps = (
                            max_pps,
                            f'Found highest PPS {format_number(max_pps)} for property {feature} '
                            f'and class {max_class}'
                        )

                if failed_features:
                    message = f'Properties and classes with PPS difference above threshold: {failed_features}'
                    return ConditionResult(ConditionCategory.FAIL, message)
                else:
                    message = overall_max_pps[1] if overall_max_pps[0] > 0 else '0 PPS found for all properties'
                    return ConditionResult(ConditionCategory.PASS, message)
            else:
                failed_features = {feature: format_number(v) for feature, v in value['train-test difference'].items()
                                   if above_threshold_fn(v)}
                if failed_features:
                    message = f'Properties with PPS difference above threshold: {failed_features}'
                    return ConditionResult(ConditionCategory.FAIL, message)
                else:
                    max_feature, max_pps = get_dict_entry_by_value(value['train-test difference'])
                    message = f'Found highest PPS {format_number(max_pps)} for property {max_feature}' \
                        if max_pps > 0 else '0 PPS found for all properties'
                    return ConditionResult(ConditionCategory.PASS, message)

        return self.add_condition(f'Train-Test properties\' Predictive Power Score difference is less than '
                                  f'{format_number(threshold)}', condition)

    def add_condition_property_pps_in_train_less_than(self: PLC, threshold: float = 0.2) -> PLC:
        """Add new condition.

        Add condition that will check that train dataset property pps is less than X. If per_class is True, the
        condition will apply per class, and a single class with pps greater than X will be enough to fail the
        condition.

        Parameters
        ----------
        threshold : float , default: 0.2
            pps upper bound

        Returns
        -------
        SFC
        """

        def condition(value: Union[Dict[Hashable, Dict[Hashable, float]],
                                   Dict[Hashable, Dict[Hashable, Dict[Hashable, float]]]]) -> ConditionResult:

            if self.per_class:
                failed_features = {}
                overall_max_pps = -np.inf, ''
                for feature, pps_info in value.items():
                    failed_classes = {class_name: format_number(pps) for class_name, pps in pps_info['train'].items()
                                      if pps >= threshold}
                    if failed_classes:
                        failed_features[feature] = failed_classes
                    # Get max diff to display when condition is passing
                    max_class, max_pps = get_dict_entry_by_value(pps_info['train'])
                    if max_pps > overall_max_pps[0]:
                        overall_max_pps = max_pps, f'Found highest PPS in train dataset {format_number(max_pps)} for ' \
                                                   f'property {feature} and class {max_class}'

                if failed_features:
                    message = f'Properties and classes in train dataset with PPS above threshold: {failed_features}'
                    return ConditionResult(ConditionCategory.FAIL, message)
                else:
                    message = overall_max_pps[1] if overall_max_pps[0] > 0 else '0 PPS found for all properties in ' \
                                                                                'train dataset'
                    return ConditionResult(ConditionCategory.PASS, message)
            else:
                failed_features = {
                    feature_name: format_number(pps_value)
                    for feature_name, pps_value in value['train'].items()
                    if pps_value >= threshold
                }

                if failed_features:
                    message = f'Properties in train dataset with PPS above threshold: {failed_features}'
                    return ConditionResult(ConditionCategory.FAIL, message)
                else:
                    max_feature, max_pps = get_dict_entry_by_value(value['train-test difference'])
                    message = (
                        f'Found highest PPS in train dataset {format_number(max_pps)} for property {max_feature}'
                        if max_pps > 0
                        else '0 PPS found for all properties in train dataset'
                    )
                    return ConditionResult(ConditionCategory.PASS, message)

        return self.add_condition(f'Train properties\' Predictive Power Score is less than '
                                  f'{format_number(threshold)}', condition)
