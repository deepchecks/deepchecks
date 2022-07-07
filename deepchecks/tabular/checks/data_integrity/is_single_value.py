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
"""Module contains is_single_value check."""
from typing import List, Union

import pandas as pd

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular.utils.messages import get_condition_passed_message
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.typing import Hashable

__all__ = ['IsSingleValue']


class IsSingleValue(SingleDatasetCheck):
    """Check if there are columns which have only a single unique value in all rows.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all
        columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based
        on columns variable.
    ignore_nan : bool, default True
        Whether to ignore NaN values in a column when counting the number of unique values.
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        ignore_nan: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.ignore_nan = ignore_nan

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value of result is a dict of all columns with number of unique values in format {column: number_of_uniques}
            display is a series with columns that have only one unique
        """
        # Validate parameters
        df = context.get_data_by_kind(dataset_kind).data
        df = select_from_dataframe(df, self.columns, self.ignore_columns)

        num_unique_per_col = df.nunique(dropna=self.ignore_nan)
        is_single_unique_value = (num_unique_per_col == 1)

        if context.with_display and is_single_unique_value.any():
            # get names of columns with one unique value
            # pylint: disable=unsubscriptable-object
            cols_with_single = is_single_unique_value[is_single_unique_value].index.to_list()
            uniques = pd.DataFrame({
                column_name: [column.sort_values(kind='mergesort').values[0]]
                for column_name, column in df.loc[:, cols_with_single].items()
            })
            uniques.index = ['Single unique value']
            display = ['The following columns have only one unique value', uniques]
        else:
            display = None

        return CheckResult(num_unique_per_col.to_dict(), header='Single Value in Column', display=display)

    def add_condition_not_single_value(self):
        """Add condition - no column contains only a single value."""
        name = 'Does not contain only a single value'

        def condition(result):
            single_value_cols = [k for k, v in result.items() if v == 1]
            if single_value_cols:
                details = f'Found {len(single_value_cols)} out of {len(result)} columns with a single value: ' \
                          f'{single_value_cols}'
                return ConditionResult(ConditionCategory.FAIL, details)
            else:
                return ConditionResult(ConditionCategory.PASS, get_condition_passed_message(result))

        return self.add_condition(name, condition)
