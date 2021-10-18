"""Module contains is_single_value check."""
from typing import Union, List
import pandas as pd
from mlchecks import SingleDatasetBaseCheck, CheckResult, validate_dataset_or_dataframe

__all__ = ['is_single_value', 'IsSingleValue']


def is_single_value(dataset: pd.DataFrame, ignore_columns: Union[str, List[str]] = None):
    """Check if there are columns which have only a single unique value in all rows.

    If found, returns column names and the single value in each of them.

    Args:
        dataset (pd.DataFrame): A Dataset object or a pd.DataFrame
        ignore_columns: single column or list of columns to exclude when checking for single values

    Returns:
        CheckResult: value is a boolean if there was at least one column with only one unique,
                     display is a series with columns that have only one unique
    """
    dataset = validate_dataset_or_dataframe(dataset)
    dataset = dataset.drop_columns_with_validation(ignore_columns)

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
        return is_single_value(dataset, ignore_columns=self.params.get('ignore_columns'))
