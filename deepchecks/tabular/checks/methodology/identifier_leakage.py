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
"""module contains Identifier Leakage check."""
from typing import Dict

import plotly.express as px

import deepchecks.ppscore as pps
from deepchecks.core import CheckResult, ConditionResult
from deepchecks.core.errors import DatasetValidationError
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.strings import format_number


__all__ = ['IdentifierLeakage']


class IdentifierLeakage(SingleDatasetCheck):
    """Check if identifiers (Index/Date) can be used to predict the label.

    Parameters
    ----------
    ppscore_params : any , default: None
        dictionary containing params to pass to ppscore predictor
    """

    def __init__(self, ppscore_params=None):
        super().__init__()
        self.ppscore_params = ppscore_params or {}

    def run_logic(self, context: Context, dataset_type: str = 'train') -> CheckResult:
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
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        dataset.assert_label()
        label_name = dataset.label_name

        relevant_columns = list(filter(None, [dataset.datetime_name, dataset.index_name, label_name]))

        if len(relevant_columns) == 1:
            raise DatasetValidationError(
                'Check is irrelevant for Datasets without index or date column'
            )

        df_pps = pps.predictors(df=dataset.data[relevant_columns], y=label_name, random_seed=42,
                                **self.ppscore_params)
        df_pps = df_pps.set_index('x', drop=True)
        s_ppscore = df_pps['ppscore']

        xaxis_layout = dict(title='Identifiers', type='category')
        yaxis_layout = dict(fixedrange=True,
                            range=(0, 1),
                            title='predictive power score (PPS)')

        red_heavy_colorscale = [
            [0, 'rgb(255, 255, 255)'],  # jan
            [0.1, 'rgb(255,155,100)'],
            [0.2, 'rgb(255, 50, 50)'],
            [0.3, 'rgb(200, 0, 0)'],
            [1, 'rgb(55, 0, 0)']
        ]

        figure = px.bar(s_ppscore, x=s_ppscore.index, y='ppscore', color='ppscore',
                        color_continuous_scale=red_heavy_colorscale)
        figure.update_layout(width=700, height=400)
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

        return CheckResult(value=s_ppscore.to_dict(), display=display)

    def add_condition_pps_not_greater_than(self, max_pps: float = 0):
        """Add condition - require columns not to have a greater pps than given max.

        Parameters
        ----------
        max_pps : float , default: 0
            Maximum allowed string length outliers ratio.
        """
        def compare_pps(result: Dict):
            not_passing_columns = {}
            for column_name in result.keys():
                score = result[column_name]
                if score > max_pps:
                    not_passing_columns[column_name] = format_number(score)
            if not_passing_columns:
                return ConditionResult(False,
                                       f'Found columns with PPS above threshold: {not_passing_columns}')
            else:
                return ConditionResult(True)

        return self.add_condition(
            f'Identifier columns PPS is not greater than {format_number(max_pps)}', compare_pps)
