"""Module contains is_single_value check."""
import pandas as pd
from mlchecks import SingleDatasetBaseCheck, CheckResult, validate_dataset_or_dataframe
from mlchecks.display import format_check_display

__all__ = ['is_single_value', 'IsSingleValue']


def is_single_value(dataset: pd.DataFrame, ignore_columns=None):
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

    is_single_unique_value = (dataset.nunique(dropna=False)==1)

    if is_single_unique_value.any():
        value = True
        # get names of columns with one unique value
        cols_with_single = is_single_unique_value[is_single_unique_value].index.to_list()
        uniques = dataset.loc[:, cols_with_single].head(1)
        uniques.index = ['Single unique value']
        display_text = f'The following columns have only one unique value<br>{uniques.to_html(index=True)}'

    else:
        value = False
        display_text = None

    html = format_check_display('Single Value in Column', is_single_value, display_text) # pylint: disable=E1136

    return CheckResult(value, display={'text/html': html})


class IsSingleValue(SingleDatasetBaseCheck):
    """Summarize given dataset parameters."""

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
