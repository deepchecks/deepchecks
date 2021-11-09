"""module contains Mixed Types check."""
from typing import Iterable, Union
import pandas as pd

import numpy as np

from mlchecks import Dataset, ensure_dataframe_type
from mlchecks.base.check import CheckResult, SingleDatasetBaseCheck, ConditionResult

__all__ = ['MixedTypes']

from mlchecks.base.dataframe_utils import filter_columns_with_validation
from mlchecks.string_utils import is_string_column, format_percent, format_columns_for_condition


class MixedTypes(SingleDatasetBaseCheck):
    """Search for various types of data in (a) column[s], including hidden mixes in strings."""

    def __init__(self, columns: Union[str, Iterable[str]] = None, ignore_columns: Union[str, Iterable[str]] = None):
        """Initialize the MixedTypes check.

        Args:
            columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored
            ones.
            ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns
            variable.
        """
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns

    def run(self, dataset, model=None) -> CheckResult:
        """Run check.

        Args:
          dataset(Dataset): Dataset to be tested.
          model: Model is ignored for this check.

        Returns:
          (CheckResult): DataFrame with rows ('strings', 'numbers') for any column with mixed types.
          numbers will also include hidden numbers in string representation.
        """
        return self._mixed_types(dataset)

    def _mixed_types(self, dataset: Union[pd.DataFrame, Dataset]) -> CheckResult:
        """Run check.

        Args:
            dataset (Dataset): Dataset to be tested.

        Returns:
            (CheckResult): DataFrame with columns('Column Name', 'Percentage') for any column that is not single typed.
        """
        # Validate parameters
        dataset: pd.DataFrame = ensure_dataframe_type(dataset)
        dataset = filter_columns_with_validation(dataset, self.columns, self.ignore_columns)

        # Result value: { Column Name: {string: pct, numbers: pct}}
        display_dict = {}

        for column_name in dataset.columns:
            column_data = dataset[column_name]
            mix = self._get_data_mix(column_data)
            if mix:
                display_dict[column_name] = mix

        df_graph = pd.DataFrame.from_dict(display_dict)
        display = df_graph if len(df_graph) > 0 else None

        return CheckResult(display_dict, check=self.__class__, display=display)

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
        nums_pct = format_percent(nums / total_rows)
        strs_pct = format_percent((np.abs(nums - total_rows)) / total_rows)

        return {'strings': strs_pct, 'numbers': nums_pct}

    def add_condition_any_type_ratio_higher_than(self, ratio: float = 0.01):
        """Add condition - Whether there are strings or numbers in any column with ratio lower than given ratio."""
        def condition(result, ratio):
            failing_columns = []
            for col, ratios in result.items():
                if ratios['strings'] < ratio or ratios['numbers'] < ratio:
                    failing_columns.append(col)
            if failing_columns:
                details = f'Columns with low type ratio: {", ".join(failing_columns)}'
                return ConditionResult(False, details)
            return ConditionResult(True)

        column_names = format_columns_for_condition(self.columns, self.ignore_columns)
        return self.add_condition(f'Any type ratio is lower than {ratio} for {column_names}', condition, ratio=ratio)
