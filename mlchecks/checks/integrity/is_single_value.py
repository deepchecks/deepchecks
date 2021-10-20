"""Module contains is_single_value check."""
from typing import Union, Iterable
import pandas as pd
from mlchecks import SingleDatasetBaseCheck, CheckResult, ensure_dataframe_type, Dataset

__all__ = ['is_single_value', 'IsSingleValue']

from mlchecks.base.dataframe_utils import filter_columns_with_validation


def is_single_value(dataset: Union[pd.DataFrame, Dataset], columns: Union[str, Iterable[str]] = None,
                       ignore_columns: Union[str, Iterable[str]] = None) -> CheckResult:
    """Check if there are columns which have only a single unique value in all rows.

    If found, returns column names and the single value in each of them.

    Args:
        dataset (pd.DataFrame): A Dataset object or a pd.DataFrame
        columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns variable

    Returns:
        CheckResult: value is a boolean if there was at least one column with only one unique,
                     display is a series with columns that have only one unique
    """
    # Validate parameters
    dataset: pd.DataFrame = ensure_dataframe_type(dataset)
    dataset = filter_columns_with_validation(dataset, columns, ignore_columns)

    is_single_unique_value = (dataset.nunique(dropna=False) == 1)

    if is_single_unique_value.any():
        value = True
        # get names of columns with one unique value
        # pylint: disable=unsubscriptable-object
        cols_with_single = is_single_unique_value[is_single_unique_value].index.to_list()
        uniques = dataset.loc[:, cols_with_single].head(1)
        uniques.index = ['Single unique value']
        display = ['The following columns have only one unique value', uniques]
    else:
        value = False
        display = None

    return CheckResult(value, header='Single Value in Column', check=is_single_value, display=display)


class IsSingleValue(SingleDatasetBaseCheck):
    """Check if there are columns which have only a single unique value in all rows."""

    def run(self, dataset, model=None) -> CheckResult:
        """
        Run is_single_value check.

        Args:
            dataset (pd.DataFrame): A Dataset object or a pd.DataFrame
            ignore_columns: list of columns to exclude when checking for single values

        Returns:
            CheckResult: value is a boolean if there was at least one column with only one unique,
            display is a series with columns that have only one unique
        """
        return is_single_value(dataset,
                               columns=self.params.get('columns'),
                               ignore_columns=self.params.get('ignore_columns'))
