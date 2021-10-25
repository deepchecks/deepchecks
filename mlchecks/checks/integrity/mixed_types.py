"""module contains Mixed Types check."""
from typing import Iterable, Union
import pandas as pd

import numpy as np

from mlchecks import Dataset, ensure_dataframe_type
from mlchecks.base.check import CheckResult, SingleDatasetBaseCheck


__all__ = ['mixed_types', 'MixedTypes']

from mlchecks.base.dataframe_utils import filter_columns_with_validation
from mlchecks.string_utils import is_string_column


def mixed_types(dataset: Union[pd.DataFrame, Dataset], columns: Union[str, Iterable[str]] = None,
                ignore_columns: Union[str, Iterable[str]] = None) -> CheckResult:
    """Search for mixed types of Data in a single column[s].

    Args:
        dataset (Dataset):
        columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns variable

    Returns:
        (CheckResult): DataFrame with columns('Column Name', 'Precentage') for any column that is not single typed.
    """
    # Validate parameters
    dataset: pd.DataFrame = ensure_dataframe_type(dataset)
    dataset = filter_columns_with_validation(dataset, columns, ignore_columns)

    # Result value: { Column Name: {string: pct, numbers: pct}}
    display_dict = {}

    for column_name in dataset.columns:
        column_data = dataset[column_name]
        mix = get_data_mix(column_data)
        if mix:
            display_dict[column_name] = mix

    df_graph = pd.DataFrame.from_dict(display_dict)

    if len(df_graph) > 0:
        display = df_graph
    else:
        display = None

    return CheckResult(df_graph, check=mixed_types, display=display)


def get_data_mix(column_data: pd.Series) -> dict:
    if is_string_column(column_data):
        return check_mixed_percentage(column_data)
    return {}


def check_mixed_percentage(column_data: pd.Series) -> dict:
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
    nums_pct = f'{nums/total_rows:.2%}'
    strs_pct = f'{(np.abs(nums-total_rows))/total_rows:.2%}'

    return {'strings': strs_pct, 'numbers': nums_pct}


class MixedTypes(SingleDatasetBaseCheck):
    """Search for various types of data in (a) coulmn[s], including hidden mixes in strings."""

    def run(self, dataset, model=None) -> CheckResult:
        """Run mixed_types.

        Args:
          dataset(Dataset):

        Returns:
          (CheckResult): DataFrame with rows ('strings', 'numbers') for any column with mixed types.
          numbers will also include hidden numbers in string representation.
        """
        return mixed_types(dataset,
                            columns=self.params.get('columns'),
                            ignore_columns=self.params.get('ignore_columns'))
