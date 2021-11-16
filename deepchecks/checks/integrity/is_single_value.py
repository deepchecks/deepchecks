"""Module contains is_single_value check."""
from typing import Union, Iterable
import pandas as pd
from deepchecks import SingleDatasetBaseCheck, CheckResult, ensure_dataframe_type, Dataset

__all__ = ['IsSingleValue']

from deepchecks.base.dataframe_utils import filter_columns_with_validation


class IsSingleValue(SingleDatasetBaseCheck):
    """Check if there are columns which have only a single unique value in all rows."""

    def __init__(self, columns: Union[str, Iterable[str]] = None, ignore_columns: Union[str, Iterable[str]] = None):
        """
        Initialize the IsSingleValue check.

        Args:
            columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored
            ones.
            ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns
            variable
        """
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns

    def run(self, dataset, model=None) -> CheckResult:
        """
        Run check.

        Args:
            dataset (Dataset): A Dataset object or a pd.DataFrame

        Returns:
            CheckResult: value is a boolean if there was at least one column with only one unique,
            display is a series with columns that have only one unique
        """
        return self._is_single_value(dataset)

    def _is_single_value(self, dataset: Union[pd.DataFrame, Dataset]) -> CheckResult:
        # Validate parameters
        dataset: pd.DataFrame = ensure_dataframe_type(dataset)
        dataset = filter_columns_with_validation(dataset, self.columns, self.ignore_columns)

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

        return CheckResult(value, header='Single Value in Column', check=self.__class__, display=display)
