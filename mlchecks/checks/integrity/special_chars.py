"""module contains Invalid Chars check."""
from collections import defaultdict
from typing import Iterable, Union
import pandas as pd
from pandas.api.types import infer_dtype

from mlchecks import Dataset, ensure_dataframe_type
from mlchecks.base.check import CheckResult, SingleDatasetBaseCheck
from mlchecks.base.dataframe_utils import filter_columns_with_validation
from mlchecks.string_utils import string_baseform

__all__ = ['special_characters', 'SpecialCharacters']


def special_characters(dataset: Union[pd.DataFrame, Dataset], columns: Union[str, Iterable[str]] = None,
                       ignore_columns: Union[str, Iterable[str]] = None, n_most_common: int = 2) -> CheckResult:
    """Search in column[s] for values that contains only special characters.

    Args:
        dataset (Dataset): a Dataset or DataFrame object.
        columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns variable
        n_most_common (int): Number of most common special-only samples to show in results
    Returns:
        (CheckResult): DataFrame with columns ('Column Name', '% Invalid Samples', 'Most Common Invalids Samples')
         for any column that contains invalid chars.
    """
    # Validate parameters
    dataset: pd.DataFrame = ensure_dataframe_type(dataset)
    dataset = filter_columns_with_validation(dataset, columns, ignore_columns)

    # Result value: { Column Name: {invalid: pct}}
    display_array = []

    for column_name in dataset.columns:
        column_data = dataset[column_name]
        # Get dict of samples to count
        special_samples = get_special_samples(column_data)
        if special_samples:
            percent = f'{sum(special_samples.values()) / column_data.size:.2%}'
            top_n_samples_items = sorted(special_samples.items(), key=lambda x: x[1], reverse=True)[:n_most_common]
            top_n_samples_values = [item[0] for item in top_n_samples_items]
            display_array.append([column_name, percent, top_n_samples_values])

    df_graph = pd.DataFrame(display_array,
                            columns=['Column Name', '% Special-Only Samples', 'Most Common Special-Only Samples'])
    df_graph = df_graph.set_index(['Column Name'])
    display = df_graph if len(df_graph) > 0 else None

    return CheckResult(df_graph, check=special_characters, display=display)


def get_special_samples(column_data: pd.Series) -> Union[dict, None]:
    if not is_stringed_type(column_data):
        return None
    samples_to_count = defaultdict(lambda: 0)
    for sample in column_data:
        if isinstance(sample, str) and len(sample) > 0 and len(string_baseform(sample)) == 0:
            samples_to_count[sample] = samples_to_count[sample] + 1

    return samples_to_count or None


def is_stringed_type(col):
    return infer_dtype(col) not in ['integer', 'decimal', 'floating']


class SpecialCharacters(SingleDatasetBaseCheck):
    """Search in column[s] for values that contains only special characters."""

    def run(self, dataset, model=None) -> CheckResult:
        """Run special_characters check.

        Args:
          dataset(Dataset):

        Returns:
          (CheckResult): DataFrame with ('invalids') for any column with special_characters chars.
        """
        return special_characters(dataset,
                                  columns=self.params.get('columns'),
                                  ignore_columns=self.params.get('ignore_columns'),
                                  n_most_common=self.params.get('n_most_common'))
