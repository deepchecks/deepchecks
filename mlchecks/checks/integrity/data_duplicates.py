"""module contains Data Duplicates check."""
from typing import List, Iterable
import pandas as pd
from pandas import DataFrame
from pandas.api.types import infer_dtype

import numpy as np

from mlchecks import Dataset
from mlchecks.base.check import CheckResult, SingleDatasetBaseCheck
from mlchecks.base.dataset import validate_dataset_or_dataframe
from mlchecks.checks.integrity.mixed_types import validate_column_list
from mlchecks.utils import MLChecksValueError


__all__ = ['data_duplicates', 'DataDuplicates']



def data_duplicates(dataset: DataFrame,
                    columns: Iterable[str] = None,
                    ignore_columns: Iterable[str] = None,
                    n_to_show: int = 5):
    """Search for Data duplicats in dataset.

    Args:
        dataset (Dataset):
        columns (List[str]): List of columns to check, if none given checks all columns Except ignored ones.
        ignore_columns (List[str]): List of columns to ignore, if none given checks based on columns variable

    Returns:
        (CheckResult):
    """
    dataset: Dataset = validate_dataset_or_dataframe(dataset)
    dataset = dataset.drop_columns_with_validation(ignore_columns)

    if columns is None:
        columns: List[str] = dataset.columns.tolist()
    else:
        columns: List[str] = validate_column_list(columns)

    n_samples = dataset.shape[0]

    if n_samples == 0:
        return CheckResult(0, header="Data Duplicates", check=data_duplicates, display=None)

    print(columns, type(columns))

    unique_data_counted = dataset[columns].groupby(columns).size()
    n_unique = len(unique_data_counted)

    percent_duplicate = 1 - (1.0 * int(n_unique)) / (1.0 * int(n_samples))

    if percent_duplicate > 0:
        unique_rows_counted = unique_data_counted.reset_index().rename(columns={0: "duplicate_count"})
        most_duplicates = unique_rows_counted[unique_rows_counted["duplicate_count"] > 1].nlargest(n_to_show, ["duplicate_count"])
        text = f'{percent_duplicate:.1%} of data are duplicates'
        display = [text, most_duplicates]
    else:
        display = None

    return CheckResult(value=percent_duplicate, header="Data Duplicates", check=data_duplicates, display=display)


class DataDuplicates(SingleDatasetBaseCheck):
    """Search for duplicate data in dataset."""

    def run(self, dataset, model=None) -> CheckResult:
        """Run data_duplicates.

        Args:
          dataset(Dataset):

        Returns:
          (CheckResult): DataFrame with rows ('strings', 'numbers') for any column with mixed types.
          numbers will also include hidden numbers in string representation.
        """
        return data_duplicates(dataset,
                            columns=self.params.get('columns'),
                            ignore_columns=self.params.get('ignore_columns'))
