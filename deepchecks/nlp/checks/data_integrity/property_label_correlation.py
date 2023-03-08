# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The property label correlation check module."""
import typing as t

import pandas as pd

import deepchecks.ppscore as pps
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.check_utils.feature_label_correlation_utils import get_pps_figure, pd_series_to_trace
from deepchecks.nlp import Context, SingleDatasetCheck
from deepchecks.nlp.task_type import TaskType
from deepchecks.tabular.utils.messages import get_condition_passed_message
from deepchecks.utils.strings import format_number
from deepchecks.utils.typing import Hashable

__all__ = ['PropertyLabelCorrelation']

PLC = t.TypeVar('PLC', bound='PropertyLabelCorrelation')

pps_url = 'https://docs.deepchecks.com/en/stable/checks_gallery/tabular/' \
          'train_test_validation/plot_feature_label_correlation_change.html'
pps_html = f'<a href={pps_url} target="_blank">Predictive Power Score</a>'


class PropertyLabelCorrelation(SingleDatasetCheck):
    """Return the PPS (Predictive Power Score) of all properties in relation to the label.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    In this check, we specifically use it to assess the ability to predict the label by a text property (e.g.
    text length, language etc.).
    A high PPS (close to 1) can mean that there's a bias in the dataset, as a single property can predict the label
    successfully, using simple classic ML algorithms - meaning that a deep learning algorithm may accidentally learn
    these properties instead of more accurate complex abstractions.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Parameters
    ----------
    ppscore_params : dict , default: None
        dictionary of additional parameters for the ppscore.predictors function
    n_top_properties : int , default: 5
        Number of properties to show, sorted by the magnitude of difference in PPS
    n_samples : int , default: 100_000
        number of samples to use for this check.
    random_state : int , default: None
        Random state for the ppscore.predictors function
    """

    def __init__(
            self,
            ppscore_params: t.Optional[t.Dict[t.Any, t.Any]] = None,
            n_top_properties: int = 5,
            n_samples: int = 100_000,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.ppscore_params = ppscore_params or {}
        self.n_top_properties = n_top_properties
        self.n_samples = n_samples

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is a dictionary with PPS per property.
            data is a bar graph of the PPS of each property.

        Raises
        ------
        DeepchecksValueError
            If the object is not a Dataset instance with a label.
        """
        text_data = context.get_data_by_kind(dataset_kind).sample(self.n_samples, random_state=context.random_state)

        label = pd.Series(text_data.label, name='label', index=text_data.index)

        # Classification labels should be of type object (and not int, for example)
        if context.task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION]:
            label = label.astype('object')

        df = text_data.properties.join(label)

        df_pps = pps.predictors(df=df, y='label', random_seed=context.random_state,
                                **self.ppscore_params)
        s_ppscore = df_pps.set_index('x', drop=True)['ppscore']

        if context.with_display:
            top_to_show = s_ppscore.head(self.n_top_properties)

            fig = get_pps_figure(per_class=False, n_of_features=len(top_to_show), x_name='property',
                                 xaxis_title='Property')
            fig.add_trace(pd_series_to_trace(top_to_show, dataset_kind.value, text_data.name))

            text = [
                'The Predictive Power Score (PPS) is used to estimate the ability of a property to predict the '
                f'label by itself (Read more about {pps_html}).'
                'A high PPS (close to 1) can mean there\'s a bias in the dataset, as a single property can predict '
                'the label successfully, meaning that the model may accidentally learn '
                'these properties instead of more accurate complex abstractions.']

            # display only if not all scores are 0
            display = [fig, *text] if s_ppscore.sum() else None
        else:
            display = None

        return CheckResult(value=s_ppscore.to_dict(), display=display, header='Property-Label Correlation')

    def add_condition_property_pps_less_than(self: PLC, threshold: float = 0.3) -> PLC:
        """
        Add condition that will check that pps of the specified properties is less than the threshold.

        Parameters
        ----------
        threshold : float , default: 0.3
            pps upper bound
        Returns
        -------
        FLC
        """

        def condition(value: t.Dict[Hashable, float]) -> ConditionResult:
            failed_properties = {
                property_name: format_number(pps_value)
                for property_name, pps_value in value.items()
                if pps_value >= threshold
            }

            if failed_properties:
                message = f'Found {len(failed_properties)} out of {len(value)} properties with PPS above threshold: ' \
                          f'{failed_properties}'
                return ConditionResult(ConditionCategory.FAIL, message)
            else:
                return ConditionResult(ConditionCategory.PASS, get_condition_passed_message(value))

        return self.add_condition(f'Properties\' Predictive Power Score is less than {format_number(threshold)}',
                                  condition)
