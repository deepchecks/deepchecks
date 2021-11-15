"""Module contains Mixed Nulls check."""
from typing import Iterable, Union

import numpy as np
import pandas as pd

from deepchecks import Dataset, CheckResult, ensure_dataframe_type
from deepchecks.base.check import SingleDatasetBaseCheck
from deepchecks.base.dataframe_utils import filter_columns_with_validation
from deepchecks.string_utils import string_baseform, format_percent
from deepchecks.utils import DeepchecksValueError

__all__ = ['MixedNulls']

DEFAULT_NULL_VALUES = {'none', 'null', 'nan', 'na', '', '\x00', '\x00\x00'}


class MixedNulls(SingleDatasetBaseCheck):
    """Search for various types of null values in a string column(s), including string representations of null."""

    def __init__(self, null_string_list: Iterable[str] = None, check_nan: bool = True,
                 columns: Union[str, Iterable[str]] = None, ignore_columns: Union[str, Iterable[str]] = None):
        """Initialize the MixedNulls check.

        Args:
            null_string_list (List[str]): List of strings to be considered alternative null representations
            check_nan(bool): Whether to add to null list to check also NaN values
            columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored
                ones.
            ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns
                variable
        """
        super().__init__()
        self.null_string_list = null_string_list
        self.check_nan = check_nan
        self.columns = columns
        self.ignore_columns = ignore_columns

    def run(self, dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            dataset (Dataset):

        Returns:
            (CheckResult): DataFrame with columns ('Column Name', 'Value', 'Count', 'Percentage') for any column which
            have more than 1 null values.
        """
        return self._mixed_nulls(dataset)

    def _validate_null_string_list(self, nsl, check_nan: bool) -> set:
        """Validate the object given is a list of strings. If null is given return default list of null values.

        Args:
            nsl: Object to validate
            check_nan (bool): Whether to add to null list to check also NaN values

        Returns:
            (set): Returns list of null values as set object
        """
        result: set
        if nsl:
            if not isinstance(nsl, Iterable):
                raise DeepchecksValueError('null_string_list must be an iterable')
            if len(nsl) == 0:
                raise DeepchecksValueError("null_string_list can't be empty list")
            if any((not isinstance(string, str) for string in nsl)):
                raise DeepchecksValueError("null_string_list must contain only items of type 'str'")
            result = set(nsl)
        else:
            # Default values
            result = set(DEFAULT_NULL_VALUES)
        if check_nan is None or check_nan is True:
            result.add(np.NaN)

        return result

    def _mixed_nulls(self, dataset: Union[pd.DataFrame, Dataset]) -> CheckResult:
        """Run check logic.

        Args:
            dataset (DataFrame): dataset to check

        Returns
            (CheckResult): DataFrame with columns ('Column Name', 'Value', 'Count', 'Fraction of data') for any column
             which have more than 1 null values.
        """
        # Validate parameters
        dataset: pd.DataFrame = ensure_dataframe_type(dataset)
        dataset = filter_columns_with_validation(dataset, self.columns, self.ignore_columns)
        null_string_list: set = self._validate_null_string_list(self.null_string_list, self.check_nan)

        # Result value
        display_array = []

        for column_name in list(dataset.columns):
            column_data = dataset[column_name]
            # TODO: Modify this once Dataset type casting mechanism is done
            if column_data.dtype != pd.StringDtype:
                continue
            # Get counts of all values in series including NaNs, in sorted order of count
            column_counts: pd.Series = column_data.value_counts(dropna=False)
            # Filter out values not in the nulls list
            keys_to_drop = [key for key in column_counts.keys() if string_baseform(key) not in null_string_list]
            null_counts = column_counts.drop(labels=keys_to_drop)
            if null_counts.size < 2:
                continue
            # Save the column info
            for key, count in null_counts.iteritems():
                display_array.append([column_name, key, count, format_percent(count / dataset.size)])

        # Create dataframe to display graph
        df_graph = pd.DataFrame(display_array, columns=['Column Name', 'Value', 'Count', 'Percent of data'])
        df_graph = df_graph.set_index(['Column Name', 'Value'])

        if len(df_graph) > 0:
            display = df_graph
        else:
            display = None

        return CheckResult(df_graph, check=self.__class__, display=display)
