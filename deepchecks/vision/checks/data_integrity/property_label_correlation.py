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
"""Module contains the property label correlation check."""

from collections import defaultdict
from typing import Any, Dict, Hashable, List, Optional, TypeVar

import pandas as pd

import deepchecks.ppscore as pps
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, DatasetKind
from deepchecks.core.check_utils.feature_label_correlation_utils import get_pps_figure, pd_series_to_trace
from deepchecks.utils.strings import format_number
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.base_checks import SingleDatasetCheck
from deepchecks.vision.context import Context
from deepchecks.vision.utils.property_label_correlation_utils import calc_properties_for_property_label_correlation
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper

__all__ = ['PropertyLabelCorrelation']

pps_url = 'https://docs.deepchecks.com/en/stable/checks_gallery/vision/' \
          'data_integrity/plot_property_label_correlation.html'
pps_html = f'<a href={pps_url} target="_blank">Predictive Power Score</a>'

PLC = TypeVar('PLC', bound='PropertyLabelCorrelation')


@docstrings
class PropertyLabelCorrelation(SingleDatasetCheck):
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
    n_top_properties: int, default: 5
        Number of features to show, sorted by the magnitude of difference in PPS
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
            min_pps_to_show: float = 0.05,
            ppscore_params: dict = None,
            n_samples: Optional[int] = 10000,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.image_properties = image_properties
        self.min_pps_to_show = min_pps_to_show
        self.n_top_properties = n_top_properties
        self.ppscore_params = ppscore_params or {}
        self._properties_results = defaultdict(list)
        self.n_samples = n_samples

    def initialize_run(self, context: Context, dataset_kind: DatasetKind):
        """Initialize run."""
        context.assert_task_type(TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION)

    def update(self, context: Context, batch: BatchWrapper, dataset_kind: DatasetKind):
        """Calculate image properties for train or test batches."""
        vision_data = context.get_data_by_kind(dataset_kind)
        data_for_properties, target = calc_properties_for_property_label_correlation(
            vision_data.task_type, batch, self.image_properties)

        self._properties_results['target'] += target

        for prop_name, property_values in data_for_properties.items():
            self._properties_results[prop_name].extend(property_values)

    def compute(self, context: Context, dataset_kind: DatasetKind) -> CheckResult:
        """Calculate the PPS between each property and the label.

        Returns
        -------
        CheckResult
            value: dictionaries of PPS values.
            display: bar graph of the PPS of each feature.
        """
        vision_data = context.get_data_by_kind(dataset_kind)
        df_props = pd.DataFrame(self._properties_results)
        # PPS task type is inferred from label dtype. For supported task types (object detection, classification),
        # the label should be regarded as categorical thus it is cast to object dtype.
        df_props['target'] = df_props['target'].apply(vision_data.label_map.get).astype('object')
        df_pps = pps.predictors(df=df_props, y='target', random_seed=context.random_state,
                                **self.ppscore_params)
        s_ppscore = df_pps.set_index('x', drop=True)['ppscore']

        if context.with_display:
            top_to_show = s_ppscore.head(self.n_top_properties)
            dataset = context.get_data_by_kind(dataset_kind)
            fig = get_pps_figure(per_class=False, n_of_features=len(top_to_show))
            fig.add_trace(pd_series_to_trace(top_to_show, dataset_kind.value, dataset.name))

            text = [
                'The Predictive Power Score (PPS) is used to estimate the ability of an image property (such as '
                'brightness)'
                f'to predict the label by itself. (Read more about {pps_html})'
                '',
                '<u>In the graph above</u>, we should suspect we have problems in our data if:',
                ''
                '<b>Train dataset PPS values are high</b>:',
                '   A high PPS (close to 1) can mean that there\'s a bias in the dataset, as a single property can '
                'predict the label successfully, using simple classic ML algorithms'
            ]

            # display only if not all scores are 0
            display = [fig, *text] if s_ppscore.sum() else None
        else:
            display = None

        return CheckResult(value=s_ppscore.to_dict(), display=display, header='Property Label Correlation')

    def add_condition_property_pps_less_than(self: PLC, threshold: float = 0.8) -> PLC:
        """
        Add condition that will check that pps of the specified properties is less than the threshold.

        Parameters
        ----------
        threshold : float , default: 0.8
            pps upper bound
        Returns
        -------
        FLC
        """

        def condition(value: Dict[Hashable, float]) -> ConditionResult:
            failed_props = {
                prop_name: format_number(pps_value)
                for prop_name, pps_value in value.items()
                if pps_value >= threshold
            }

            if failed_props:
                message = f'Found {len(failed_props)} out of {len(value)} properties with PPS above threshold: ' \
                          f'{failed_props}'
                return ConditionResult(ConditionCategory.FAIL, message)
            else:
                return ConditionResult(ConditionCategory.PASS, 'Passed for all of the properties')

        return self.add_condition(f'Properties\' Predictive Power Score is less than {format_number(threshold)}',
                                  condition)
