"""Module contains Mixed Nulls check."""
from collections import defaultdict
from typing import Iterable, Union, Dict, List

import numpy as np
import pandas as pd

from mlchecks import Dataset, CheckResult, ensure_dataframe_type
from mlchecks.base.check import SingleDatasetBaseCheck, ConditionResult
from mlchecks.base.dataframe_utils import filter_columns_with_validation
from mlchecks.string_utils import string_baseform, format_percent
from mlchecks.utils import MLChecksValueError

__all__ = ['mixed_nulls', 'MixedNulls']

DEFAULT_NULL_VALUES = {'none', 'null', 'nan', 'na', '', '\x00', '\x00\x00'}


def validate_null_string_list(nsl, check_nan: bool) -> set:
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
            raise MLChecksValueError('null_string_list must be an iterable')
        if len(nsl) == 0:
            raise MLChecksValueError("null_string_list can't be empty list")
        if any((not isinstance(string, str) for string in nsl)):
            raise MLChecksValueError("null_string_list must contain only items of type 'str'")
        result = set(nsl)
    else:
        # Default values
        result = set(DEFAULT_NULL_VALUES)
    if check_nan is None or check_nan is True:
        result.add(np.NaN)

    return result


def mixed_nulls(dataset: Union[pd.DataFrame, Dataset], null_string_list: Iterable[str] = None, check_nan: bool = True,
                columns: Union[str, Iterable[str]] = None, ignore_columns: Union[str, Iterable[str]] = None) \
        -> CheckResult:
    """Search for various types of null values in a string column(s), including string representations of null.

    Args:
        dataset (DataFrame): dataset to check
        null_string_list (List[str]): List of strings to be considered alternative null representations
        check_nan(bool): Whether to add to null list to check also NaN values
        columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns variable

    Returns
        (CheckResult): Dictionary of columns having more than 1 null type in form:
        `{ column: { null_value: { count: <x>, percent: <y>} } }`
    """
    # Validate parameters
    dataset: pd.DataFrame = ensure_dataframe_type(dataset)
    dataset = filter_columns_with_validation(dataset, columns, ignore_columns)
    null_string_list: set = validate_null_string_list(null_string_list, check_nan)

    # Result value
    display_array = []
    result_dict = defaultdict(dict)

    for column_name in list(dataset.columns):
        column_data = dataset[column_name]
        # TODO: Modify this once Dataset type casting mechanism is done
        if column_data.dtype != pd.StringDtype:
            continue
        # Get counts of all values in series including NaNs, in sorted order of count
        column_counts: pd.Series = column_data.value_counts(dropna=False)
        # Filter out values not in the nulls list
        null_counts = {value: count for value, count in column_counts.items()
                       if string_baseform(value) in null_string_list}
        if len(null_counts) < 2:
            continue
        # Save the column info
        for null_value, count in null_counts.items():
            percent = count / dataset.size
            display_array.append([column_name, null_value, count, format_percent(percent)])
            result_dict[column_name][null_value] = {'count': count, 'percent': percent}

    # Create dataframe to display table
    if display_array:
        df_graph = pd.DataFrame(display_array, columns=['Column Name', 'Value', 'Count', 'Percent of data'])
        df_graph = df_graph.set_index(['Column Name', 'Value'])
        display = df_graph
    else:
        display = None

    return CheckResult(result_dict, check=mixed_nulls, display=display)


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
                           ignore_columns=self.params.get('ignore_columns'),
                           columns=self.params.get('columns'),
                           check_nan=self.params.get('check_nan'))

    def add_condition_max_different_nulls(self, max_nulls: int, columns: List[str] = None,
                                          ignore_columns: List[str] = None):
        """Add condition that a column have a maximum number of different null values.

        Args:
            max_nulls (int): Maximum number allowed of different null values.
            columns (List[str]): Column to limit the condition to. If none, runs on all.
            ignore_columns (List[str]):
        """
        if columns and not ignore_columns:
            column_names = f'columns: {",".join(columns)}'
        elif ignore_columns and not columns:
            column_names = f'all columns ignoring: {",".join(ignore_columns)}'
        elif not columns and not ignore_columns:
            column_names = 'all columns'
        else:
            raise MLChecksValueError('Can not define columns and ignore_columns together')

        def condition(result: Dict) -> ConditionResult:
            columns_in_result = set(result.keys())
            if columns:
                columns_in_result = columns_in_result & set(columns)
            if ignore_columns:
                columns_in_result = columns_in_result - set(ignore_columns)
            not_passing_columns = []
            for column in columns_in_result:
                nulls = result[column]
                num_nulls = len(nulls)
                if num_nulls > max_nulls:
                    not_passing_columns.append(column)
            if not_passing_columns:
                not_passing_columns = ', '.join(not_passing_columns)
                return ConditionResult(False,
                                       f'Found columns {not_passing_columns} with more than {max_nulls} null types')
            else:
                return ConditionResult(True)

        self.add_condition(f'No more than {max_nulls} null types for {column_names}', condition)
        return self
