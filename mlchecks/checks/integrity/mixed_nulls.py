"""Module contains Mixed Nulls check."""
from typing import List, Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame, StringDtype

from mlchecks import Dataset, CheckResult, validate_dataset_or_dataframe, validate_column
from mlchecks.base.check import SingleDatasetBaseCheck
from mlchecks.utils import MLChecksValueError

__all__ = ['mixed_nulls', 'MixedNulls']
DEFAULT_NULL_VALUES = {'none', 'null', 'nan', 'na', '', "\x00", "\x00\x00"}
SPECIAL_CHARS: str = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~\n"


def validate_null_string_list(nsl) -> set:
    """Validate the object given is a list of strings. If null is given return default list of null values.

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
        return DEFAULT_NULL_VALUES


def string_baseform(string: str):
    """Remove special characters from given string.

    Args:
        string (str): string to remove special characters from

    Returns:
        (str): string without special characters
    """
    if not isinstance(string, str):
        return string
    return string.translate(str.maketrans('', '', SPECIAL_CHARS)).lower()


def mixed_nulls(dataset: DataFrame, null_string_list: Iterable[str] = None, column: str = None) -> CheckResult:
    """Search for various types of null values in a string column(s), including string representations of null.

    Args:
        dataset (Dataset):
        null_string_list (List[str]): List of strings to be considered alternative null representations
        column(str): Single column to check. if none given checks all the string columns

    Returns
        (CheckResult): DataFrame with columns ('Column Name', 'Value', 'Count', 'Percentage') for any column which have
        more than 1 null values.
    """
    # Validate parameters
    dataset: Dataset = validate_dataset_or_dataframe(dataset)
    null_string_list: set = validate_null_string_list(null_string_list)
    null_string_list.add(np.NaN)
    columns: List[str] = validate_column(column, dataset)

    # Result value
    display_array = []

    for column_name in columns:
        column_data = dataset[column_name]
        # TODO: Modify this once Dataset type casting mechanism is done
        if column_data.dtype != StringDtype:
            continue
        # Get counts of all values in series including NaNs, in sorted order of count
        column_counts: pd.Series = column_data.value_counts(dropna=False)
        # Filter out values not in the nulls list
        keys_to_drop = [key for key in column_counts.keys() if string_baseform(key) not in null_string_list]
        null_counts = column_counts.drop(labels=keys_to_drop)
        if null_counts.size < 2:
            continue
        # Get the dominant (the series is sorted), and drop it
        # dominant_null = null_counts.index[0], null_counts.iloc[0]
        # other_nulls_counts = null_counts.drop(null_counts.index[0])
        # Save the column info
        for key, count in null_counts.iteritems():
            display_array.append([column_name, key, count, count / dataset.size])

    # Create dataframe to display graph
    df_graph = pd.DataFrame(display_array, columns=['Column Name', 'Value', 'Count', 'Percentage'])

    return CheckResult(df_graph, display={'text/html': df_graph.to_html()})


class MixedNulls(SingleDatasetBaseCheck):
    """Search for various types of null values in a string column(s), including string representations of null."""

    def run(self, dataset, model=None) -> CheckResult:
        """Run mix_nulls.

        Args:
            dataset (Dataset):

        Returns:
            (CheckResult): DataFrame with columns ('Column Name', 'Value', 'Count', 'Percentage') for any column which
            have more than 1 null values.
        """
        return mixed_nulls(dataset,
                           null_string_list=self.params.get('null_string_list'),
                           column=self.params.get('column'))
