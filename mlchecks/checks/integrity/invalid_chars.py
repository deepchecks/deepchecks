"""module contains Invalid Chars check."""
from typing import List, Iterable
import pandas as pd
from pandas import DataFrame
from pandas.api.types import infer_dtype


from mlchecks import Dataset
from mlchecks.base.check import CheckResult, SingleDatasetBaseCheck
from mlchecks.base.dataset import validate_dataset_or_dataframe
from mlchecks.checks.integrity.string_utils import string_baseform
from mlchecks.display import format_check_display
from mlchecks.utils import validate_column_list


__all__ = ['invalid_chars', 'InvalidChars']


def invalid_chars(dataset: DataFrame, columns: Iterable[str]=None, ignore_columns: Iterable[str]=None ) -> CheckResult:
    """Search for invalid chars in column[s].

    Args:
        dataset (Dataset):
        columns (List[str]): List of columns to check, if none given checks all columns Except ignored ones.
        ignore_columns (List[str]): List of columns to ignore, if none given checks based on columns variable

    Returns:
        (CheckResult): DataFrame with columns('Column Name', 'Precentage') for any column that contains invalid chars.
    """
    # Validate parameters
    dataset: Dataset = validate_dataset_or_dataframe(dataset)
    dataset = dataset.drop_columns_with_validation(ignore_columns)
    dataset = dataset.keep_only_columns_with_validation(columns)
    # Result value: { Column Name: {invalid: pct}}
    display_array = []

    for column_name in columns:
        try:
            column_data = dataset[column_name]
        except KeyError: #Column is not in data
            continue
        inv = get_invalid_chars(column_data)
        if inv is not None:
            display_array.append([column_name,inv])

    df_graph = pd.DataFrame(display_array, columns=['Column Name', '% Invalid Samples'])

    visual = df_graph.to_html(index=False, justify='left') if len(df_graph) > 0 else None
    formatted_html = format_check_display('Invalid Chars', invalid_chars, visual)
    return CheckResult(df_graph, display={'text/html': formatted_html})


def get_invalid_chars(column_data: pd.Series) -> str :
    if is_stringed_type(column_data):
        return check_invalid_chars(column_data)
    return None


def is_stringed_type(col):
    return infer_dtype(col) not in ['integer', 'decimal', 'floating']


def check_invalid_chars(column_data: pd.Series) -> str:
    total_rows = column_data.count()

    def is_invalid_char(x):
        return string_baseform(x) != x

    invalids = sum(column_data.apply(is_invalid_char))
    if invalids == 0:
        return None

    # Then we've got a mix
    invalids_pct = f'{invalids/total_rows:.2%}'

    return invalids_pct


class InvalidChars(SingleDatasetBaseCheck):
    """Search for invalid chars in (a) coulmn[s]."""

    def run(self, dataset, model=None) -> CheckResult:
        """Run invalid_chars.

        Args:
          dataset(Dataset):

        Returns:
          (CheckResult): DataFrame with ('invalids') for any column with invalid chars.
        """
        return invalid_chars(dataset,
                            columns=self.params.get('columns'),
                            ignore_columns=self.params.get('ignore_columns'))
