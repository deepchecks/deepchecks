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
"""module contains Data Duplicates check."""
from typing import Union, List

import pandas as pd

from deepchecks import Dataset, ensure_dataframe_type
from deepchecks.base.check import CheckResult, SingleDatasetBaseCheck, ConditionResult
from deepchecks.utils.dataframes import filter_columns_with_validation
from deepchecks.utils.strings import format_percent
from deepchecks.utils.typing import Hashable
from deepchecks.errors import DeepchecksValueError


__all__ = ['DataDuplicates']


class DataDuplicates(SingleDatasetBaseCheck):
    """Search for duplicate data in dataset.

    Args:
        columns (Hashable, Iterable[Hashable]):
            List of columns to check, if none given checks
            all columns Except ignored ones.
        ignore_columns (Hashable, Iterable[Hashable]):
            List of columns to ignore, if none given checks
            based on columns variable.
        n_to_show (int):
            number of most common duplicated samples to show.
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        n_to_show: int = 5
    ):
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_to_show = n_to_show

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            dataset(Dataset): any dataset.

        Returns:
            (CheckResult): percentage of duplicates and display of the top n_to_show most duplicated.
        """
        df: pd.DataFrame = ensure_dataframe_type(dataset)
        df = filter_columns_with_validation(df, self.columns, self.ignore_columns)

        data_columns = list(df.columns)

        n_samples = df.shape[0]

        if n_samples == 0:
            raise DeepchecksValueError('Dataset does not contain any data')

        group_unique_data = df[data_columns].groupby(data_columns, dropna=False).size()
        n_unique = len(group_unique_data)

        percent_duplicate = 1 - (1.0 * int(n_unique)) / (1.0 * int(n_samples))

        if percent_duplicate > 0:
            # patched for anonymous_series
            is_anonymous_series = 0 in group_unique_data.keys().names
            if is_anonymous_series:
                new_name = str(group_unique_data.keys().names)
                new_index = group_unique_data.keys()
                new_index.names = [new_name if name==0 else name for name in new_index.names]
                group_unique_data = group_unique_data.reindex(new_index)
            duplicates_counted = group_unique_data.reset_index().rename(columns={0: 'Number of Duplicates'})
            if is_anonymous_series:
                duplicates_counted.rename(columns={new_name: 0}, inplace=True)

            most_duplicates = duplicates_counted[duplicates_counted['Number of Duplicates'] > 1]. \
                nlargest(self.n_to_show, ['Number of Duplicates'])

            most_duplicates = most_duplicates.set_index('Number of Duplicates')

            text = f'{format_percent(percent_duplicate)} of data samples are duplicates'
            display = [text, most_duplicates]
        else:
            display = None

        return CheckResult(value=percent_duplicate, display=display)

    def add_condition_ratio_not_greater_than(self, max_ratio: float = 0):
        """Add condition - require duplicate ratio to not surpass max_ratio.

        Args:
            max_ratio (float): Maximum ratio of duplicates.
        """
        def max_ratio_condition(result: float) -> ConditionResult:
            if result > max_ratio:
                return ConditionResult(False, f'Found {format_percent(result)} duplicate data')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Duplicate data is not greater than {format_percent(max_ratio)}',
                                  max_ratio_condition)
