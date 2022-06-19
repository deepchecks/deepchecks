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
"""module contains Data Duplicates check."""
from typing import List, Union

import numpy as np

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DatasetValidationError
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.strings import format_list, format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['DataDuplicates']


class DataDuplicates(SingleDatasetCheck):
    """Checks for duplicate samples in the dataset.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        List of columns to check, if none given checks
        all columns Except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        List of columns to ignore, if none given checks
        based on columns variable.
    n_to_show : int , default: 5
        number of most common duplicated samples to show.
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        n_to_show: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_to_show = n_to_show

    def run_logic(self, context: Context, dataset_kind):
        """Run check.

        Returns
        -------
        CheckResult
            percentage of duplicates and display of the top n_to_show most duplicated.
        """
        df = context.get_data_by_kind(dataset_kind).data
        df = select_from_dataframe(df, self.columns, self.ignore_columns)

        data_columns = list(df.columns)

        n_samples = df.shape[0]

        if n_samples == 0:
            raise DatasetValidationError('Dataset does not contain any data')

        # HACK: pandas have bug with groupby on category dtypes, so until it fixed, change dtypes manually
        category_columns = df.dtypes[df.dtypes == 'category'].index.tolist()
        if category_columns:
            df = df.astype({c: 'object' for c in category_columns})

        group_unique_data = df[data_columns].groupby(data_columns, dropna=False).size()
        n_unique = len(group_unique_data)

        percent_duplicate = 1 - (1.0 * int(n_unique)) / (1.0 * int(n_samples))

        if context.with_display and percent_duplicate > 0:
            # patched for anonymous_series
            is_anonymous_series = 0 in group_unique_data.keys().names
            if is_anonymous_series:
                new_name = str(group_unique_data.keys().names)
                new_index = group_unique_data.keys()
                new_index.names = [new_name if name == 0 else name for name in new_index.names]
                group_unique_data = group_unique_data.reindex(new_index)
            duplicates_counted = group_unique_data.reset_index().rename(columns={0: 'Number of Duplicates'})
            if is_anonymous_series:
                duplicates_counted.rename(columns={new_name: 0}, inplace=True)

            most_duplicates = duplicates_counted[duplicates_counted['Number of Duplicates'] > 1]. \
                nlargest(self.n_to_show, ['Number of Duplicates'])

            indexes = []
            for row in most_duplicates.iloc():
                indexes.append(format_list(df.index[np.all(df == row[data_columns], axis=1)].to_list()))

            most_duplicates['Instances'] = indexes

            most_duplicates = most_duplicates.set_index(['Instances', 'Number of Duplicates'])

            text = f'{format_percent(percent_duplicate)} of data samples are duplicates. '
            explanation = 'Each row in the table shows an example of duplicate data and the number of times it appears.'
            display = [text, explanation, most_duplicates]

        else:
            display = None

        return CheckResult(value=percent_duplicate, display=display)

    def add_condition_ratio_less_or_equal(self, max_ratio: float = 0):
        """Add condition - require duplicate ratio to be less or equal to max_ratio.

        Parameters
        ----------
        max_ratio : float , default: 0
            Maximum ratio of duplicates.
        """
        def max_ratio_condition(result: float) -> ConditionResult:
            details = f'Found {format_percent(result)} duplicate data'
            category = ConditionCategory.PASS if result <= max_ratio else ConditionCategory.WARN
            return ConditionResult(category, details)

        return self.add_condition(f'Duplicate data ratio is less or equal to {format_percent(max_ratio)}',
                                  max_ratio_condition)
