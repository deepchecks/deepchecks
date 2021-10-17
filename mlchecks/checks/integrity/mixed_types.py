"""module contains Mixed Types check."""
from typing import List, Iterable
import pandas as pd
from pandas import DataFrame
from pandas.api.types import infer_dtype

import numpy as np

from mlchecks import Dataset
from mlchecks.base.check import CheckResult, SingleDatasetBaseCheck
from mlchecks.base.dataset import validate_dataset_or_dataframe
from mlchecks.display import format_check_display
from mlchecks.utils import validate_column_list


__all__ = ['mixed_types', 'MixedTypes']




def mixed_types(dataset: DataFrame, columns: Iterable[str]=None, ignore_columns: Iterable[str]=None ) -> CheckResult:
    """Search for mixed types of Data in a single column[s].

    Args:
        dataset (Dataset):
        columns (List[str]): List of columns to check, if none given checks all columns Except ignored ones.
        ignore_columns (List[str]): List of columns to ignore, if none given checks based on columns variable

    Returns:
        (CheckResult): DataFrame with columns('Column Name', 'Precentage') for any column that is not single typed.
    """
    # Validate parameters
    dataset: Dataset = validate_dataset_or_dataframe(dataset)
    dataset = dataset.drop_columns_with_validation(ignore_columns)
    if columns is None:
        columns: List[str] = dataset.columns
    else:
        columns: set = validate_column_list(columns)

    # Result value: { Column Name: {string: pct, numbers: pct}}
    display_dict = {}

    for column_name in columns:
        try:
            column_data = dataset[column_name]
        except KeyError: #Column is not in data
            continue
        mix = get_data_mix(column_data)
        if mix:
            display_dict[column_name] = mix

    df_graph = pd.DataFrame.from_dict(display_dict)

    visual = df_graph.to_html() if len(df_graph) > 0 else None
    formatted_html = format_check_display('Mixed Types', mixed_types, visual)
    return CheckResult(df_graph, display={'text/html': formatted_html})


def get_data_mix(column_data: pd.Series) -> dict :
    if is_mixed_type(column_data):
        try:
            #string casted numbers are castable, so if we fail this the column is mixed.
            pd.to_numeric(column_data)
        except ValueError:
            #column is truely mixed.
            return check_mixed_percentage(column_data)
    return {}

def is_mixed_type(col):
    # infer_dtype returns a 'mixed[-xxx]' even if real value is number
    # if it returns string, we still need to validate no hidden numbers present
    return infer_dtype(col) in ['mixed', 'mixed-integer', 'string']

def check_mixed_percentage(column_data: pd.Series) -> dict:
    total_rows = column_data.count()

    def is_float(x):
        try:
            float(x)
            return True
        except:
            return False

    nums = sum(column_data.apply(is_float))
    if nums == total_rows or nums == 0:
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
