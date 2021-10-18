"""module contains Invalid Chars check."""
from typing import Iterable
import pandas as pd
from pandas import DataFrame
from pandas.api.types import infer_dtype

from mlchecks import Dataset
from mlchecks.base.check import CheckResult, SingleDatasetBaseCheck
from mlchecks.base.dataset import validate_dataset_or_dataframe
from mlchecks.checks.integrity.string_utils import string_baseform
from mlchecks.utils import MLChecksValueError

__all__ = ['invalid_characters', 'InvalidCharacters']


def invalid_characters(dataset: DataFrame, columns: Iterable[str] = None, ignore_columns: Iterable[str] = None) \
    -> CheckResult:
    """Search in column[s] for values that contains only special characters.

    Args:
        dataset (Dataset): a Dataset or DataFrame object.
        columns (List[str]): List of columns to check, if none given checks all columns Except ignored ones.
        ignore_columns (List[str]): List of columns to ignore, if none given checks based on columns variable

    Returns:
        (CheckResult): DataFrame with columns('Column Name', 'Percentage') for any column that contains invalid chars.
    """
    # Validate parameters
    dataset: Dataset = validate_dataset_or_dataframe(dataset)
    common = set(columns or []).intersection(set(ignore_columns or []))
    if common:
        raise MLChecksValueError(f'Same column can not appear in "columns" and "ignore_columns": {", ".join(common)}')
    dataset = dataset.drop_columns_with_validation(ignore_columns)
    dataset = dataset.keep_only_columns_with_validation(columns)
    # Result value: { Column Name: {invalid: pct}}
    display_array = []

    for column_name in dataset.columns:
        try:
            column_data = dataset[column_name]
        except KeyError: #Column is not in data
            continue
        inv = get_invalid_chars(column_data)
        if inv is not None:
            display_array.append([column_name,inv])

    df_graph = pd.DataFrame(display_array, columns=['Column Name', '% Invalid Samples'])
    display = df_graph if len(df_graph) > 0 else None

    return CheckResult(df_graph, header='Invalid Characters', check=invalid_characters, display=display)


def get_invalid_chars(column_data: pd.Series) -> str :
    if is_stringed_type(column_data):
        return check_invalid_chars(column_data)
    return None


def is_stringed_type(col):
    return infer_dtype(col) not in ['integer', 'decimal', 'floating']


def check_invalid_chars(column_data: pd.Series) -> str:
    total_rows = column_data.count()

    def is_all_invalid_char(x):
        return isinstance(x, str) and len(x) > 0 and len(string_baseform(x)) == 0

    invalids = sum(column_data.apply(is_all_invalid_char))
    if invalids == 0:
        return None

    # Then we've got a mix
    invalids_pct = f'{invalids/total_rows:.2%}'

    return invalids_pct


class InvalidCharacters(SingleDatasetBaseCheck):
    """Search in column[s] for values that contains only special characters."""

    def run(self, dataset, model=None) -> CheckResult:
        """Run invalid_characters check.

        Args:
          dataset(Dataset):

        Returns:
          (CheckResult): DataFrame with ('invalids') for any column with invalid chars.
        """
        return invalid_characters(dataset,
                                  columns=self.params.get('columns'),
                                  ignore_columns=self.params.get('ignore_columns'))
