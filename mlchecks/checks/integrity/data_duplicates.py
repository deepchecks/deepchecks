"""module contains Data Duplicates check."""
from typing import Iterable, Union

import pandas as pd

from mlchecks import Dataset, ensure_dataframe_type
from mlchecks.base.check import CheckResult, SingleDatasetBaseCheck
from mlchecks.base.dataframe_utils import filter_columns_with_validation
from mlchecks.utils import MLChecksValueError

__all__ = ['data_duplicates', 'DataDuplicates']


def data_duplicates(dataset: Union[pd.DataFrame, Dataset], columns: Union[str, Iterable[str]] = None,
                    ignore_columns: Union[str, Iterable[str]] = None, n_to_show: int = 5) -> CheckResult:
    """Search for Data duplicates in dataset.

    Args:
        dataset (DataFrame or Dataset): any dataset.
        columns (str, Iterable[str]): List of columns to check, if none given checks all columns Except ignored ones.
        ignore_columns (str, Iterable[str]): List of columns to ignore, if none given checks based on columns variable.
        n_to_show (int): number of most duplicated to show.
    Returns:
        (CheckResult): percentage of duplicates and display of the top n_to_show most duplicated.

    Raises:
        MLChecksValueError: If the dataset is empty or columns not in dataset.
    """
    dataset: pd.DataFrame = ensure_dataframe_type(dataset)
    dataset = filter_columns_with_validation(dataset, columns, ignore_columns)

    data_columns = list(dataset.columns)

    n_samples = dataset.shape[0]

    if n_samples == 0:
        raise MLChecksValueError('Dataset does not contain any data')

    group_unique_data = dataset[data_columns].groupby(data_columns).size()
    n_unique = len(group_unique_data)

    percent_duplicate = 1 - (1.0 * int(n_unique)) / (1.0 * int(n_samples))

    if percent_duplicate > 0:
        duplicates_counted = group_unique_data.reset_index().rename(columns={0: 'duplicate_count'})
        most_duplicates = duplicates_counted[duplicates_counted['duplicate_count'] > 1]. \
            nlargest(n_to_show if n_to_show is not None else 5, ['duplicate_count'])
        text = f'{percent_duplicate:.1%} of data are duplicates'
        display = [text, most_duplicates]
    else:
        display = None

    return CheckResult(value=percent_duplicate, header='Data Duplicates', check=data_duplicates, display=display)


class DataDuplicates(SingleDatasetBaseCheck):
    """Search for duplicate data in dataset."""

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """Run data_duplicates.

        Args:
          dataset(Dataset): any dataset.

        Returns:
          (CheckResult): percentage of duplicates and display of the top n_to_show most duplicated.
        """
        return data_duplicates(dataset,
                               columns=self.params.get('columns'),
                               ignore_columns=self.params.get('ignore_columns'),
                               n_to_show=self.params.get('n_to_show'))
