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
"""Module contains is_single_value check."""
from typing import Union, List

from deepchecks.base.check_context import CheckRunContext
from deepchecks import SingleDatasetBaseCheck, CheckResult, ConditionResult
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.typing import Hashable


__all__ = ['IsSingleValue']


class IsSingleValue(SingleDatasetBaseCheck):
    """Check if there are columns which have only a single unique value in all rows.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all
        columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based
        on columns variable.
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None
    ):
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns

    def run_logic(self, context: CheckRunContext, dataset_type: str = 'train') -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is a boolean if there was at least one column with only one unique,
            display is a series with columns that have only one unique
        """
        # Validate parameters
        if dataset_type == 'train':
            df = context.train.data
        else:
            df = context.test.data

        df = select_from_dataframe(df, self.columns, self.ignore_columns)

        is_single_unique_value = (df.nunique(dropna=False) == 1)

        if is_single_unique_value.any():
            # get names of columns with one unique value
            # pylint: disable=unsubscriptable-object
            cols_with_single = is_single_unique_value[is_single_unique_value].index.to_list()
            value = list(cols_with_single)
            uniques = df.loc[:, cols_with_single].head(1)
            uniques.index = ['Single unique value']
            display = ['The following columns have only one unique value', uniques]
        else:
            value = None
            display = None

        return CheckResult(value, header='Single Value in Column', display=display)

    def add_condition_not_single_value(self):
        """Add condition - not single value."""
        name = 'Does not contain only a single value'

        def condition(result):
            if result:
                return ConditionResult(False, f'Found columns with a single value: {result}')
            return ConditionResult(True)

        return self.add_condition(name, condition)
