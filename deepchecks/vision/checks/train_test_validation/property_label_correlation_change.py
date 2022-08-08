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
"""Module contains the simple feature distribution check."""
from collections import defaultdict
from typing import Any, Dict, Hashable, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_float_dtype

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.check_utils.feature_label_correlation_utils import (get_feature_label_correlation,
                                                                         get_feature_label_correlation_per_class)
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import ModelValidationError
from deepchecks.utils.dict_funcs import get_dict_entry_by_value
from deepchecks.utils.strings import format_number
from deepchecks.vision import Context, TrainTestCheck
from deepchecks.vision.batch_wrapper import Batch
from deepchecks.vision.utils.image_functions import crop_image
from deepchecks.vision.utils.image_properties import default_image_properties
from deepchecks.vision.utils.vision_properties import PropertiesInputType
from deepchecks.vision.vision_data import TaskType

__all__ = ['PropertyLabelCorrelationChange']

pps_url = 'https://docs.deepchecks.com/en/stable/checks_gallery/vision/' \
          'train_test_validation/plot_feature_label_correlation_change.html'
pps_html = f'<a href={pps_url} target="_blank">Predictive Power Score</a>'


FLC = TypeVar('FLC', bound='PropertyLabelCorrelationChange')


# FeatureLabelCorrelationChange
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
        Each property is dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of:
        - 'numeric' - for continuous ordinal outputs.
        - 'categorical' - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        For more on image / label properties, see the :ref:`property guide </user-guide/vision/vision_properties.rst>`
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
    """

    def __init__(
            self,
            image_properties: Optional[List[Dict[str, Any]]] = None,
            n_top_properties: int = 3,
            per_class: bool = True,
            random_state: int = None,
            min_pps_to_show: float = 0.05,
            ppscore_params: dict = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.image_properties = image_properties if image_properties else default_image_properties

        self.min_pps_to_show = min_pps_to_show
        self.per_class = per_class
        self.n_top_properties = n_top_properties
        self.random_state = random_state
        self.ppscore_params = ppscore_params or {}

        self._train_properties = defaultdict(list)
        self._test_properties = defaultdict(list)
        self._train_properties['target'] = []
        self._test_properties['target'] = []

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
        """Calculate image properties for train or test batches."""
        dataset = context.get_data_by_kind(dataset_kind)

        if dataset_kind == DatasetKind.TRAIN:
            properties_results = self._train_properties
        else:
            properties_results = self._test_properties

        imgs = []
        target = []

        if dataset.task_type == TaskType.OBJECT_DETECTION:
            for img, labels in zip(batch.images, batch.labels):
                for label in labels:
                    label = label.cpu().detach().numpy()
                    bbox = label[1:]
                    cropped_img = crop_image(img, *bbox)
                    if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                        continue
                    class_id = int(label[0])
                    imgs += [cropped_img]
                    target += [dataset.label_id_to_name(class_id)]
            property_type = PropertiesInputType.BBOXES
        else:
            for img, classes_ids in zip(batch.images, dataset.get_classes(batch.labels)):
                imgs += [img] * len(classes_ids)
                target += list(map(dataset.label_id_to_name, classes_ids))
            property_type = PropertiesInputType.IMAGES

        properties_results['target'] += target

        data_for_properties = batch.vision_properties(imgs, self.image_properties, property_type)

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

        # PPS task type is inferred from label dtype. For most computer vision tasks, it's safe to assume that unless
        # the label is a float, then the task type is not regression and thus the label is cast to object dtype.
        # For the known task types (object detection, classification), classification is always selected.
        col_dtype = 'object'
        if context.train.task_type == TaskType.OTHER:
            if self.is_float_column(df_train['target']) or self.is_float_column(df_test['target']):
                col_dtype = 'float'
        elif context.train.task_type not in (TaskType.OBJECT_DETECTION, TaskType.CLASSIFICATION):
            raise ModelValidationError(
                f'Check must be explicitly adopted to the new task type {context.train.task_type}, so that the '
                f'label type used by the PPS predictor would be appropriate.')

        df_train['target'] = df_train['target'].astype(col_dtype)
        df_test['target'] = df_test['target'].astype(col_dtype)

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
                                                                         random_state=self.random_state,
                                                                         with_display=context.with_display)
        else:
            ret_value, display = get_feature_label_correlation(df_train,
                                                               'target',
                                                               df_test,
                                                               'target',
                                                               self.ppscore_params,
                                                               self.n_top_properties,
                                                               min_pps_to_show=self.min_pps_to_show,
                                                               random_state=self.random_state,
                                                               with_display=context.with_display)

        if display:
            display += text

        return CheckResult(value=ret_value, display=display, header='Feature Label Correlation Change')

    @staticmethod
    def is_float_column(col: pd.Series) -> bool:
        """Check if a column must be a float - meaning does it contain fractions.

        Parameters
        ----------
        col : pd.Series
            The column to check.

        Returns
        -------
        bool
            True if the column is float, False otherwise.
        """
        if not is_float_dtype(col):
            return False

        return (col.round() != col).any()

    def add_condition_property_pps_difference_less_than(self: FLC, threshold: float = 0.2,
                                                        include_negative_diff: bool = False) -> FLC:
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

    def add_condition_property_pps_in_train_less_than(self: FLC, threshold: float = 0.2) -> FLC:
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
