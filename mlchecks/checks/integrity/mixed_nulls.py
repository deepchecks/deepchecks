from typing import List, Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame, StringDtype

from mlchecks import Dataset, CheckResult
from mlchecks.base.check import SingleDatasetBaseCheck
from mlchecks.utils import MLChecksValueError, validate_dataset

__all__ = ['mixed_nulls']


def validate_null_string_list(nsl) -> set:
    """Validate the object given is a list of strings. If null is given return default list of null values

    Args:
        nsl: Object to validate

    Returns:
        (set): Returns list of null values as set object
    """
    if nsl:
        if not isinstance(nsl, Iterable):
            raise MLChecksValueError('null_string_list must be an iterable')
        if len(nsl) == 0:
            raise MLChecksValueError("null_string_list can't be empty list")
        if any((not isinstance(string, str) for string in nsl)):
            raise MLChecksValueError("null_string_list must contain only items of type 'str'")
        return set(nsl)
    else:
        # Default values
        return {'none', 'null', 'nan', 'na', '', "\x00", "\x00\x00"}


def string_baseform(string: str):
    """Remove special characters from given string

    Args
        string (str): string to remove special characters from

    Returns
        (str): string without special characters
    """
    SPECIAL_CHARS: str = " !\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\n"
    if not isinstance(string, str):
        return string
    return string.translate(str.maketrans('', '', SPECIAL_CHARS))


def validate_column(column, dataset) -> List[str]:
    """If column is not None, make sure it exists in the datasets, and return list column name.
    If column is None return list of all columns in the dataset

    Args:
        column ([None, str]): column name or None
        dataset (Dataset): Dataset working on

    Returns:
        (List[str]): List with column names to work on
    """
    if column is None:
        # If column is None works on all columns
        return [dataset.columns]
    else:
        if not isinstance(column, str):
            raise MLChecksValueError(f"column type must be 'None' or 'str' but got: {type(column).__name__}")
        if len(column) == 0:
            raise MLChecksValueError("column can't be empty string")
        if column not in dataset.columns:
            raise MLChecksValueError(f"column {column} isn't found in the dataset")
        return [column]


def mixed_nulls(dataset: DataFrame, null_string_list: Iterable[str] = None, column: str = None) -> CheckResult:
    """The check searches for various types of null values in a string column(s), including string representations of
    null.

    Args:
        dataset (Dataset):
        null_string_list (List[str]): List of strings to be considered alternative null representations
        column(str): Single column to check. if none given checks all the string columns

    Returns
        (CheckResult): Value is dictionary with all columns that have at least 2 null representations, in the following
        format: { 'column name' : { 'dominant_null': ['value', count], 'other_nulls': {'value2': count2, ...} }
    """
    # Validate parameters
    dataset: Dataset = validate_dataset(dataset)
    null_string_list: set = validate_null_string_list(null_string_list)
    null_string_list.add(np.NaN)
    columns: List[str] = validate_column(column, dataset)

    # Result value
    column_to_nulls = {}
    display_array = []

    for column_name in columns:
        column_data = dataset[column_name]
        # TODO if the user explicitly asked for column and it's not a string, throw error?
        if column_data.dtype != StringDtype:
            continue
        # Get counts of all values in series including NaNs, in sorted order of count
        column_counts: pd.Series = column_data.value_counts(dropna=False)
        # Filter out values not in the nulls list
        keys_to_drop = [key for key in column_counts.keys() if string_baseform(key) in null_string_list]
        null_counts = column_counts.drop(labels=keys_to_drop)
        if null_counts.size < 2:
            continue
        # Get the dominant (the series is sorted), and drop it
        dominant_null = null_counts.index[0], null_counts.iloc[0]
        other_nulls_counts = null_counts.drop(null_counts.index[0])

        # Save the column info
        column_to_nulls[column_name] = {
            'dominant_null': dominant_null,
            'other_nulls': other_nulls_counts.to_dict()
        }
        for key, count in null_counts.iteritems():
            display_array.append([column_name, key, count, count / dataset.size])

    # Create dataframe to display graph
    df_graph = pd.DataFrame(display_array, columns=['Column Name', 'Value', 'Count', 'Percentage'])

    return CheckResult(column_to_nulls, display={'text/html': df_graph.to_html()})


class MixNulls(SingleDatasetBaseCheck):
    """
    """
    def run(self, dataset, model=None) -> CheckResult:
        return mixed_nulls(dataset,
                           null_string_list=self.params.get('null_string_list'),
                           column=self.params.get('column'))
