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
"""module contains the Identifier-Label Correlation check."""
from typing import Dict

import pandas as pd
import plotly.express as px

import deepchecks.ppscore as pps
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DatasetValidationError
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular.dataset import _get_dataset_docs_tag
from deepchecks.tabular.utils.messages import get_condition_passed_message
from deepchecks.utils.strings import format_number

__all__ = ['IdentifierLabelCorrelation']


class IdentifierLabelCorrelation(SingleDatasetCheck):
    """Check if identifiers (Index/Date) can be used to predict the label.

    Parameters
    ----------
    ppscore_params : any , default: None
        dictionary containing params to pass to ppscore predictor
    """

    def __init__(self, ppscore_params=None, **kwargs):
        super().__init__(**kwargs)
        self.ppscore_params = ppscore_params or {}

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is a dictionary with PPS per feature column.
            data is a bar graph of the PPS of each feature.

        Raises
        ------
        DeepchecksValueError
            If the object is not a Dataset instance with a label.
        """
        dataset = context.get_data_by_kind(dataset_kind)
        label_name = dataset.label_name

        relevant_data = pd.DataFrame({
            it.name: it
            for it in (dataset.index_col, dataset.datetime_col, dataset.label_col)
            if it is not None
        })

        if len(relevant_data.columns) == 1:
            raise DatasetValidationError(
                'Dataset does not contain an index or a datetime',
                html=f'Dataset does not contain an index or a datetime. see {_get_dataset_docs_tag()}'
            )

        df_pps = pps.predictors(
            df=relevant_data,
            y=label_name,
            random_seed=42,
            **self.ppscore_params
        )

        df_pps = df_pps.set_index('x', drop=True)
        s_ppscore = df_pps['ppscore']

        if context.with_display:
            xaxis_layout = dict(
                title='Identifiers',
                type='category',
                # NOTE:
                # the range, in this case, is needed to fix a problem with
                # too wide bars when there are only one or two of them`s on
                # the plot, plus it also centralizes them`s on the plot
                # The min value of the range (range(min. max)) is bigger because
                # otherwise bars will not be centralized on the plot, they will
                # appear on the left part of the plot (that is probably because of zero)
                range=(-3, len(s_ppscore.index) + 2)
            )
            yaxis_layout = dict(
                fixedrange=True,
                range=(0, 1),
                title='predictive power score (PPS)'
            )

            red_heavy_colorscale = [
                [0, 'rgb(255, 255, 255)'],  # jan
                [0.1, 'rgb(255,155,100)'],
                [0.2, 'rgb(255, 50, 50)'],
                [0.3, 'rgb(200, 0, 0)'],
                [1, 'rgb(55, 0, 0)']
            ]

            figure = px.bar(s_ppscore, x=s_ppscore.index, y='ppscore', color='ppscore',
                            color_continuous_scale=red_heavy_colorscale)
            figure.update_layout(
                height=400
            )
            figure.update_layout(
                dict(
                    xaxis=xaxis_layout,
                    yaxis=yaxis_layout,
                    coloraxis=dict(
                        cmin=0,
                        cmax=1
                    )
                )
            )

            text = ['The PPS represents the ability of a feature to single-handedly predict another feature or label.',
                    'For Identifier columns (Index/Date) PPS should be nearly 0, otherwise date and index have some '
                    'predictive effect on the label.']

            # display only if not all scores are 0
            display = [figure, *text] if s_ppscore.sum() else None
        else:
            display = None

        return CheckResult(value=s_ppscore.to_dict(), display=display)

    def add_condition_pps_less_or_equal(self, max_pps: float = 0):
        """Add condition - require columns' pps to be less or equal to threshold.

        Parameters
        ----------
        max_pps : float , default: 0
            Maximum allowed string length outliers ratio.
        """
        def compare_pps(result: Dict):
            not_passing_columns = {k: format_number(score) for k, score in result.items() if score > max_pps}
            if not_passing_columns:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Found {len(not_passing_columns)} out of {len(result)} columns with PPS above'
                                       f' threshold: {not_passing_columns}')
            else:
                return ConditionResult(ConditionCategory.PASS, get_condition_passed_message(result))

        return self.add_condition(
            f'Identifier columns PPS is less or equal to {format_number(max_pps)}', compare_pps)
