# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains Mixed Nulls check."""
import math
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular.utils.messages import get_condition_passed_message
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.features import N_TOP_MESSAGE
from deepchecks.utils.strings import format_percent, string_baseform
from deepchecks.utils.typing import Hashable

__all__ = ['MixedNulls']


DEFAULT_NULL_VALUES = {'none', 'null', 'nan', 'na', '', '\x00', '\x00\x00'}


class MixedNulls(SingleDatasetCheck):
    """Search for various types of null values, including string representations of null.

    Parameters
    ----------
    null_string_list : Iterable[str] , default: None
        List of strings to be considered alternative null representations
    check_nan : bool , default: True
        Whether to add to null list to check also NaN values
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable
    n_top_columns : int , optional
        amount of columns to show ordered by feature importance (date, index, label are first)
    """

    def __init__(
        self,
        null_string_list: Iterable[str] = None,
        check_nan: bool = True,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        n_top_columns: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.null_string_list = null_string_list
        self.check_nan = check_nan
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_columns = n_top_columns

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            Value is dict with columns as key, and dict of null values as value:
            {column: {null_value: {count: x, percent: y}, ...}, ...}
            display is DataFrame with columns ('Column Name', 'Value', 'Count', 'Percentage') for any column that
            has more than 1 null values.
        """
        dataset = context.get_data_by_kind(dataset_kind)
        df = dataset.data

        df = select_from_dataframe(df, self.columns, self.ignore_columns)
        null_string_list = self._validate_null_string_list(self.null_string_list)

        # Result value
        display_array = []
        result_dict = {}

        for column_name in list(df.columns):
            column_data = df[column_name]
            if is_categorical_dtype(column_data) is True:
                # NOTE:
                # 'pandas.Series.value_counts' and 'pandas.Series.apply'
                # work in an unusual way with categorical data types
                # - 'value_counts' returns all categorical values even if they are not in series
                # - 'apply' applies function to each category, not to values
                # therefore we processing categorical dtypes differently
                # NOTE:
                # 'Series.value_counts' method transforms null values like 'None', 'pd.Na', 'pd.NaT'
                # into 'np.nan' therefore it cannot be used for usual dtypes, because we will lose info
                # about all different null types in the column
                null_counts = {}
                for value, count in column_data.value_counts(dropna=False).to_dict().items():
                    if count > 0:
                        if pd.isna(value):
                            null_counts[nan_type(value)] = count
                        elif string_baseform(value) in null_string_list:
                            null_counts[repr(value).replace('\'', '"')] = count
            else:
                string_null_counts = {
                    repr(value).replace('\'', '"'): count
                    for value, count in column_data.value_counts(dropna=True).iteritems()
                    if string_baseform(value) in null_string_list
                }
                nan_data_counts = column_data[column_data.isna()].apply(nan_type).value_counts().to_dict()
                null_counts = {**string_null_counts, **nan_data_counts}

            result_dict[column_name] = {}
            # Save the column nulls info
            for null_value, count in null_counts.items():
                percent = count / len(column_data)
                display_array.append([column_name, null_value, count, format_percent(percent)])
                result_dict[column_name][null_value] = {'count': count, 'percent': percent}

        # Create dataframe to display table
        if context.with_display and display_array:
            df_graph = pd.DataFrame(display_array, columns=['Column Name', 'Value', 'Count', 'Percent of data'])
            order = df_graph['Column Name'].value_counts(ascending=False).index[:self.n_top_columns]
            df_graph = df_graph.set_index(['Column Name', 'Value'])
            df_graph = df_graph.loc[order, :]
            display = [N_TOP_MESSAGE % self.n_top_columns, df_graph]
        else:
            display = None

        return CheckResult(result_dict, display=display)

    def _validate_null_string_list(self, nsl) -> set:
        """Validate the object given is a list of strings. If null is given return default list of null values.

        Parameters
        ----------
        nsl
            Object to validate

        Returns
        -------
        set
            Returns list of null values as set object
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

        return result

    def add_condition_different_nulls_less_equal_to(self, max_allowed_null_types: int = 1):
        """Add condition - require column's number of different null values to be less or equal to threshold.

        Parameters
        ----------
        max_allowed_null_types : int , default: 1
            Number of different null value types which is the maximum allowed.
        """
        def condition(result: Dict) -> ConditionResult:
            not_passing_columns = [k for k, v in result.items() if len(v) > max_allowed_null_types]
            if not_passing_columns:
                details = f'Found {len(not_passing_columns)} out of {len(result)} columns with amount of null types ' \
                          f'above threshold: {not_passing_columns}'
                return ConditionResult(ConditionCategory.FAIL, details)
            else:
                return ConditionResult(ConditionCategory.PASS, get_condition_passed_message(result))

        return self.add_condition(f'Number of different null types is less or equal to {max_allowed_null_types}',
                                  condition)


def nan_type(x):
    if x is np.nan:
        return 'numpy.nan'
    elif x is pd.NA:
        return 'pandas.NA'
    elif x is pd.NaT:
        return 'pandas.NaT'
    elif isinstance(x, float) and math.isnan(x):
        return 'math.nan'
    return str(x)
