"""module contains Columns Info check."""
from typing import Iterable, Union

import pandas as pd

from mlchecks import Dataset, ensure_dataframe_type
from mlchecks.base.check import CheckResult, SingleDatasetBaseCheck
from mlchecks.base.dataframe_utils import filter_columns_with_validation
from mlchecks.utils import MLChecksValueError
from mlchecks.string_utils import format_percent


__all__ = ['data_duplicates', 'DataDuplicates']


def data_duplicates(dataset: Union[pd.DataFrame, Dataset], columns: Union[str, Iterable[str]] = None,
                    ignore_columns: Union[str, Iterable[str]] = None, n_to_show: int = 5) -> CheckResult:
    """Search for Data duplicates in dataset.

    Args:
        dataset (DataFrame or Dataset): any dataset. Default behavior should be to run only on dataset features.
                                        Columns can override that and ignore_columns can filter features out.
        columns (str, Iterable[str]): List of columns to check, if none given checks all columns Except ignored ones.
        ignore_columns (str, Iterable[str]): List of columns to ignore, if none given checks based on columns variable.
        n_to_show (int): number of most common duplicated samples to show.
    Returns:
        (CheckResult): percentage of duplicates and display of the top n_to_show most common duplicated samples.

    Raises:
        MLChecksValueError: If the dataset is empty or columns not in dataset.
    """
    df: pd.DataFrame = ensure_dataframe_type(dataset)
    df = filter_columns_with_validation(df, columns, ignore_columns)

    data_columns = list(df.columns)

    n_samples = df.shape[0]

    if n_samples == 0:
        raise MLChecksValueError('Dataset does not contain any data')

    group_unique_data = df[data_columns].groupby(data_columns).size()
    n_unique = len(group_unique_data)

    percent_duplicate = 1 - (1.0 * int(n_unique)) / (1.0 * int(n_samples))

    if percent_duplicate > 0:
        duplicates_counted = group_unique_data.reset_index().rename(columns={0: 'Number of Duplicates'})
        most_duplicates = duplicates_counted[duplicates_counted['Number of Duplicates'] > 1]. \
            nlargest(n_to_show, ['Number of Duplicates'])

        most_duplicates = most_duplicates.set_index('Number of Duplicates')

        text = f'{format_percent(percent_duplicate)} of data samples are duplicates'
        display = [text, most_duplicates]
    else:
        display = None

    return CheckResult(value=percent_duplicate, check=data_duplicates, display=display)


class DataDuplicates(SingleDatasetBaseCheck):
    """Search for duplicate data in dataset."""

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """Run data_duplicates.

        Args:
          dataset(Dataset): any dataset.

        Returns:
          (CheckResult): percentage of duplicates and display of the top n_to_show most duplicated.
        """
        n_to_show = self.params.get('n_to_show') if self.params.get('n_to_show') is not None else 5
        return data_duplicates(dataset,
                               columns=self.params.get('columns'),
                               ignore_columns=self.params.get('ignore_columns'),
                               n_to_show=n_to_show)
