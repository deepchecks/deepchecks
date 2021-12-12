# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""module contains Identifier Leakage check."""
from typing import Union, Dict

import pandas as pd

import deepchecks.ppscore as pps
from deepchecks import Dataset
from deepchecks.base.check import CheckResult, SingleDatasetBaseCheck, ConditionResult
from deepchecks.utils.plot import create_colorbar_barchart_for_check
from deepchecks.utils.strings import format_percent
from deepchecks.errors import DeepchecksValueError


__all__ = ['IdentifierLeakage']


class IdentifierLeakage(SingleDatasetBaseCheck):
    """Check if identifiers (Index/Date) can be used to predict the label.

    Args:
        ppscore_params: dictionary containing params to pass to ppscore predictor
    """

    def __init__(self, ppscore_params=None):
        super().__init__()
        self.ppscore_params = ppscore_params

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
          dataset(Dataset): any dataset.
          model: ignored in check (default: None).

        Returns:
            (CheckResult):
                value is a dictionary with PPS per feature column.
                data is a bar graph of the PPS of each feature.

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance with a label
        """
        return self._identifier_leakage(dataset)

    def _identifier_leakage(self, dataset: Union[pd.DataFrame, Dataset], ppscore_params=None) -> CheckResult:
        Dataset.validate_dataset(dataset)
        dataset.validate_label()
        ppscore_params = ppscore_params or {}

        relevant_columns = list(filter(None, [dataset.date_name, dataset.index_name, dataset.label_name]))

        if len(relevant_columns) == 1:
            raise DeepchecksValueError('Dataset needs to have a date or index column.')

        df_pps = pps.predictors(df=dataset.data[relevant_columns], y=dataset.label_name, random_seed=42,
                                **ppscore_params)
        df_pps = df_pps.set_index('x', drop=True)
        s_ppscore = df_pps['ppscore']

        def plot():
            # Create graph:
            create_colorbar_barchart_for_check(x=s_ppscore.index, y=s_ppscore.values,
                                               ylabel='predictive power score (PPS)',
                                               xlabel='Identifiers', color_map='gist_heat_r', color_shift_midpoint=0.1,
                                               color_label='PPS', check_name=self.__class__.__name__)

        text = ['The PPS represents the ability of a feature to single-handedly predict another feature or label.',
                'For Identifier columns (Index/Date) PPS should be nearly 0, otherwise date and index have some '
                'predictive effect on the label.']

        return CheckResult(value=s_ppscore.to_dict(), display=[plot, *text])

    def add_condition_pps_not_greater_than(self, max_pps: float = 0):
        """Add condition - require columns not to have a greater pps than given max.

        Args:
            max_pps (int): Maximum allowed string length outliers ratio.
        """
        def compare_pps(result: Dict):
            not_passing_columns = []
            for column_name in result.keys():
                score = result[column_name]
                if score > max_pps:
                    not_passing_columns.append(column_name)
            if not_passing_columns:
                not_passing_str = ', '.join(map(str, not_passing_columns))
                return ConditionResult(False,
                                       f'Found columns with greater pps than {format_percent(max_pps)}: '
                                       f'{not_passing_str}')
            else:
                return ConditionResult(True)

        return self.add_condition(
            f'Identifier columns do not have a greater pps than {format_percent(max_pps)}',
            compare_pps)
