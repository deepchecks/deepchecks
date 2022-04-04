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
from typing import Callable, TypeVar, Hashable, Dict, Union

import numpy as np
import pandas as pd

from deepchecks import ConditionResult
from deepchecks.core import CheckResult, DatasetKind
from deepchecks.core.check_utils.single_feature_contribution_utils import get_single_feature_contribution, \
    get_single_feature_contribution_per_class
from deepchecks.core.condition import ConditionCategory
from deepchecks.utils.strings import format_number
from deepchecks.vision import Context, TrainTestCheck
from deepchecks.vision.batch_wrapper import Batch
from deepchecks.vision.utils.image_properties import default_image_properties, validate_properties
from deepchecks.vision.utils.image_functions import crop_image
from deepchecks.vision.vision_data import TaskType

__all__ = ['SimpleFeatureContribution']

pps_url = 'https://docs.deepchecks.com/en/stable/examples/vision/' \
          'checks/methodology/simple_feature_contribution' \
          '.html?utm_source=display_output&utm_medium=referral&utm_campaign=check_link'
pps_html = f'<a href={pps_url} target="_blank">Predictive Power Score</a>'

SFC = TypeVar('SFC', bound='SimpleFeatureContribution')


class SimpleFeatureContribution(TrainTestCheck):
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
        representing attributes of said method. 'output_type' must be one of 'continuous'/'discrete'
    per_class : bool, default: True
        boolean that indicates whether the results of this check should be calculated for all classes or per class in
        label. If True, the conditions will be run per class as well.
    n_top_properties: int, default: 5
        Number of features to show, sorted by the magnitude of difference in PPS
    ppscore_params: dict, default: None
        dictionary of additional parameters for the ppscore predictor function
    """

    def __init__(
            self,
            image_properties: Dict[str, Callable] = None,
            n_top_properties: int = 3,
            per_class: bool = True,
            ppscore_params: dict = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if image_properties:
            validate_properties(image_properties)
            self.image_properties = image_properties
        else:
            self.image_properties = default_image_properties

        self.per_class = per_class
        self.n_top_properties = n_top_properties
        self.ppscore_params = ppscore_params or {}

        self._train_properties = defaultdict(list)
        self._test_properties = defaultdict(list)
        self._train_properties['target'] = []
        self._test_properties['target'] = []

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
        """Calculate image properties for train or test batches."""
        if dataset_kind == DatasetKind.TRAIN:
            dataset = context.train
            properties = self._train_properties
        else:
            dataset = context.test
            properties = self._test_properties

        imgs = []
        target = []

        if dataset.task_type == TaskType.OBJECT_DETECTION:
            for img, labels in zip(batch.images, batch.labels):
                for label in labels:
                    label = np.array(label)
                    bbox = label[1:]
                    cropped_img = crop_image(img, *bbox)
                    if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                        continue
                    class_id = label[0]
                    imgs += [cropped_img]
                    target += [class_id]
        else:
            for img, classes_ids in zip(batch.images, dataset.get_classes(batch.labels)):
                imgs += [img] * len(classes_ids)
                target += classes_ids

        properties['target'] += target

        for single_property in self.image_properties:
            properties[single_property['name']].extend(single_property['method'](imgs))

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
            ret_value, display = get_single_feature_contribution_per_class(df_train,
                                                                           'target',
                                                                           df_test,
                                                                           'target',
                                                                           self.ppscore_params,
                                                                           self.n_top_properties)
        else:
            ret_value, display = get_single_feature_contribution(df_train,
                                                                 'target',
                                                                 df_test,
                                                                 'target',
                                                                 self.ppscore_params,
                                                                 self.n_top_properties)

        if display:
            display += text

        return CheckResult(value=ret_value, display=display, header='Simple Feature Contribution')

    def add_condition_feature_pps_difference_not_greater_than(self: SFC, threshold: float = 0.2) -> SFC:
        """Add new condition.

        Add condition that will check that difference between train
        dataset property pps and test dataset property pps is not greater than X. If per_class is True, the condition
        will apply per class, and a single class with pps difference greater than X will be enough to fail the
        condition.

        Parameters
        ----------
        threshold : float , default: 0.2
            train test ps difference upper bound.

        Returns
        -------
        SFC
        """

        def condition(value: Dict[Hashable, Dict[Hashable, float]]) -> ConditionResult:
            if self.per_class is True:
                failed_features = {
                    feature_name: format_number(pps_value)
                    for feature_name, pps_value in
                    zip(value.keys(), [max(value[f]['train-test difference'].values()) for f in value.keys()])
                    if pps_value > threshold
                }

            else:
                failed_features = {
                    feature_name: format_number(pps_value)
                    for feature_name, pps_value in value['train-test difference'].items()
                    if pps_value > threshold
                }

            if failed_features:
                message = f'Features with PPS difference above threshold: {failed_features}'
                return ConditionResult(ConditionCategory.FAIL, message)
            else:
                return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(f'Train-Test properties\' Predictive Power Score difference is not greater than '
                                  f'{format_number(threshold)}', condition)

    def add_condition_feature_pps_in_train_not_greater_than(self: SFC, threshold: float = 0.2) -> SFC:
        """Add new condition.

        Add condition that will check that train dataset property pps is not greater than X. If per_class is True, the
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
            if self.per_class is True:
                failed_features = {
                    feature_name: format_number(pps_value)
                    for feature_name, pps_value in
                    zip(value.keys(), [max(value[f]['train'].values()) for f in value.keys()])
                    if pps_value > threshold
                }

            else:
                failed_features = {
                    feature_name: format_number(pps_value)
                    for feature_name, pps_value in value['train'].items()
                    if pps_value > threshold
                }

            if failed_features:
                message = f'Features in train dataset with PPS above threshold: {failed_features}'
                return ConditionResult(ConditionCategory.FAIL, message)
            else:
                return ConditionResult(ConditionCategory.PASS)

        return self.add_condition(f'Train properties\' Predictive Power Score is not greater than '
                                  f'{format_number(threshold)}', condition)
