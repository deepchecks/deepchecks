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
"""module contains Mixed Types check."""
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.features import N_TOP_MESSAGE, column_importance_sorter_df
from deepchecks.utils.strings import format_list, format_number, format_percent, get_ellipsis, is_string_column
from deepchecks.utils.typing import Hashable

__all__ = ['MixedDataTypes']


class MixedDataTypes(SingleDatasetCheck):
    """Detect columns which contain a mix of numerical and string values.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns
        except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns
        variable.
    n_top_columns : int , optional
        amount of columns to show ordered by feature importance (date, index, label are first)
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        n_top_columns: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_columns = n_top_columns

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is a dict where the key is the column name as key and the value is the ratio 'strings' and 'numbers'
            for any column with mixed data types.
            numbers will also include hidden numbers in string representation.
        """
        dataset = context.get_data_by_kind(dataset_kind)
        feature_importance = context.feature_importance

        df = select_from_dataframe(dataset.data, self.columns, self.ignore_columns)

        # Result value: { Column Name: {string: pct, numbers: pct}}
        display_dict = {}
        result_dict = {}

        for column_name in df.columns:
            column_data = df[column_name].dropna()
            mix = self._get_data_mix(column_data)
            result_dict[column_name] = mix
            if context.with_display and mix:
                # Format percents for display
                formated_mix = {}
                formated_mix['Strings'] = format_percent(mix['strings'])
                formated_mix['Numbers'] = format_percent(mix['numbers'])
                formated_mix['Strings examples'] = [get_ellipsis(strr, 15) for strr in mix['strings_examples']]
                formated_mix['Numbers examples'] = '[' + format_list([format_number(float(num))
                                                                      for num in mix['numbers_examples']]) + ']'
                display_dict[column_name] = formated_mix

        if display_dict:
            df_graph = pd.DataFrame.from_dict(display_dict)
            df_graph = column_importance_sorter_df(df_graph.T, dataset, feature_importance,
                                                   self.n_top_columns).T
            display = [N_TOP_MESSAGE % self.n_top_columns, df_graph]
        else:
            display = None

        return CheckResult(result_dict, display=display)

    def _get_data_mix(self, column_data: pd.Series) -> dict:
        if is_string_column(column_data):
            return self._check_mixed_percentage(column_data)
        return {}

    def _check_mixed_percentage(self, column_data: pd.Series) -> dict:
        total_rows = column_data.count()

        numbers_in_col = set()
        strings_in_col = set()

        def is_float(x) -> bool:
            try:
                float(x)
                if len(numbers_in_col) < 3:
                    numbers_in_col.add(x)
                return True
            except ValueError:
                if len(strings_in_col) < 3:
                    strings_in_col.add(x)
                return False

        nums = sum(column_data.apply(is_float))
        if nums in (total_rows, 0):
            return {}

        # Then we've got a mix
        nums_pct = nums / total_rows
        strs_pct = (np.abs(nums - total_rows)) / total_rows

        return {'strings': strs_pct, 'numbers': nums_pct,
                'strings_examples': strings_in_col, 'numbers_examples': numbers_in_col}

    def add_condition_rare_type_ratio_not_in_range(self, ratio_range: Tuple[float, float] = (0.01, 0.1)):
        """Add condition - Whether the ratio of rarer data type (strings or numbers) is not in the "danger zone".

        The "danger zone" represents the following logic - if the rarer data type is, for example, 30% of the data,
        than the column is presumably supposed to contain both numbers and string values. If the rarer data type is,
        for example, less than 1% of the data, than it's presumably a contamination, but a negligible one. In the range
        between, there is a real chance that the rarer data type may represent a problem to model training and
        inference.

        Parameters
        ----------
        ratio_range : Tuple[float, float] , default: (0.01 , 0.1)
            The range between which the ratio of rarer data type in the column is
            considered a problem.
        """
        def condition(result):
            no_mix_columns = []
            failing_columns = []
            for col, ratios in result.items():
                # Columns without a mix contains empty dict for ratios
                if not ratios:
                    no_mix_columns.append(col)
                    continue
                rarer_ratio = min(ratios['strings'], ratios['numbers'])
                if ratio_range[0] < rarer_ratio < ratio_range[1]:
                    failing_columns.append(col)
            if failing_columns:
                details = f'Found {len(failing_columns)} out of {len(result)} columns with non-negligible quantities ' \
                          f'of samples with a different data type from the majority of samples: {failing_columns}'
                return ConditionResult(ConditionCategory.WARN, details)
            details = f'{len(result)} columns passed: found {len(result) - len(no_mix_columns)} columns with ' \
                      f'negligible types mix, and {len(no_mix_columns)} columns without any types mix'
            return ConditionResult(ConditionCategory.PASS, details)

        name = f'Rare data types in column are either more than {format_percent(ratio_range[1])} or less ' \
               f'than {format_percent(ratio_range[0])} of the data'
        return self.add_condition(name, condition)
