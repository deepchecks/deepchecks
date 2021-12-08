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
"""module contains Mixed Types check."""
from typing import List, Union
import pandas as pd

import numpy as np

from deepchecks import Dataset, ensure_dataframe_type
from deepchecks.base.check import CheckResult, SingleDatasetBaseCheck, ConditionResult
from deepchecks.utils.dataframes import filter_columns_with_validation
from deepchecks.utils.features import calculate_feature_importance_or_null, column_importance_sorter_df
from deepchecks.utils.strings import is_string_column, format_percent, format_columns_for_condition
from deepchecks.utils.typing import Hashable


__all__ = ['MixedTypes']


class MixedTypes(SingleDatasetBaseCheck):
    """Search for various types of data in (a) column[s], including hidden mixes in strings.

    Args:
        columns (Union[Hashable, List[Hashable]]):
            Columns to check, if none are given checks all columns
            except ignored ones.
        ignore_columns (Union[Hashable, List[Hashable]]):
            Columns to ignore, if none given checks based on columns
            variable.
        n_top_columns (int): (optinal - used only if model was specified)
          amount of columns to show ordered by feature importance (date, index, label are first)
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        n_top_columns: int = 10
    ):
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_columns = n_top_columns

    def run(self, dataset, model=None) -> CheckResult:
        """Run check.

        Args:
          dataset(Dataset): Dataset to be tested.
          model: Model is ignored for this check.

        Returns:
          (CheckResult): DataFrame with rows ('strings', 'numbers') for any column with mixed types.
          numbers will also include hidden numbers in string representation.
        """
        feature_importances = calculate_feature_importance_or_null(dataset, model)
        return self._mixed_types(dataset, feature_importances)

    def _mixed_types(self, dataset: Union[pd.DataFrame, Dataset], feature_importances: pd.Series=None) -> CheckResult:
        """Run check.

        Args:
            dataset (Dataset): Dataset to be tested.

        Returns:
            (CheckResult): DataFrame with columns('Column Name', 'Percentage') for any column that is not single typed.
        """
        # Validate parameters
        original_dataset = dataset
        dataset: pd.DataFrame = ensure_dataframe_type(dataset)
        dataset = filter_columns_with_validation(dataset, self.columns, self.ignore_columns)

        # Result value: { Column Name: {string: pct, numbers: pct}}
        display_dict = {}
        result_dict = {}

        for column_name in dataset.columns:
            column_data = dataset[column_name].dropna()
            mix = self._get_data_mix(column_data)
            if mix:
                result_dict[column_name] = mix
                # Format percents for display
                display_dict[column_name] = {k: format_percent(v) for k, v in mix.items()}

        if display_dict:
            df_graph = pd.DataFrame.from_dict(display_dict)
            df_graph = column_importance_sorter_df(df_graph.T, original_dataset, feature_importances,
                                                   self.n_top_columns).T
            display = df_graph
        else:
            display = None

        return CheckResult(result_dict, display=display)

    def _get_data_mix(self, column_data: pd.Series) -> dict:
        if is_string_column(column_data):
            return self._check_mixed_percentage(column_data)
        return {}

    def _check_mixed_percentage(self, column_data: pd.Series) -> dict:
        total_rows = column_data.count()

        def is_float(x):
            try:
                float(x)
                return True
            except ValueError:
                return False

        nums = sum(column_data.apply(is_float))
        if nums in (total_rows, 0):
            return {}

        # Then we've got a mix
        nums_pct = nums / total_rows
        strs_pct = (np.abs(nums - total_rows)) / total_rows

        return {'strings': strs_pct, 'numbers': nums_pct}

    def add_condition_rare_type_ratio_not_less_than(self, max_rare_type_ratio: float = 0.01):
        """Add condition - Whether the rarer data type (strings or numbers) have ratio higher than given ratio.

        Args:
            max_rare_type_ratio (float): Minimal ratio allowed for the rarer type (numbers or strings)
        """
        def condition(result, max_rare_type_ratio):
            failing_columns = []
            for col, ratios in result.items():
                if ratios['strings'] < max_rare_type_ratio or ratios['numbers'] < max_rare_type_ratio:
                    failing_columns.append(col)
            if failing_columns:
                details = f'Found columns with low type ratio: {", ".join(map(str, failing_columns))}'
                return ConditionResult(False, details)
            return ConditionResult(True)

        column_names = format_columns_for_condition(self.columns, self.ignore_columns)
        name = f'Rare type ratio is not less than {format_percent(max_rare_type_ratio)} of samples in {column_names}'
        return self.add_condition(name, condition_func=condition, max_rare_type_ratio=max_rare_type_ratio)
