"""Module contains is_single_value check."""
from mlchecks import SingleDatasetBaseCheck, CheckResult, validate_dataset
import pandas as pd
from sklearn.base import BaseEstimator
from mlchecks.utils import model_type_validation

__all__ = ['is_single_value', 'IsSingleValue']


def is_single_value(dataset: pd.DataFrame, ignore_columns=None):
    """
    Check if there are columns which have only a single unique value in all samples.
    If so returns column names and the value.

    Args:
        dataset (pd.DataFrame): A Dataset object or a pd.DataFrame 
        ignore_columns: list of columns to exclude when checking for single values

    Returns:
        CheckResult: value is a boolean if there was at least one column with only one unique,
                     display is a series with columns that have only one unique
    """
    dataset = validate_dataset(dataset)

    if ignore_columns:
        dataset = dataset.drop(ignore_columns, axis='columns')
    
    is_single_unique_value = (dataset.nunique(dropna=False)==1)
    
    if is_single_unique_value.any():
        value = True
        # get names of columns with one unique value
        cols_with_single = is_single_unique_value[is_single_unique_value].index.to_list()
        uniques = dataset.loc[:, cols_with_single].head(1)
        uniques.index = ['Single unique value']
        # uniques = pd.concat([pd.DataFrame({'Column Name:':['Single Unique Value:']}), uniques], axis='columns')
        # uniques.name = 'columns_with_single_value'
        html = f'Notice - Some columns have only one unique value:<br>{uniques.to_html(index=True)}'

    else:
        value = False    
        html = f'Result OK - all checked columns have more than one unique value'


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