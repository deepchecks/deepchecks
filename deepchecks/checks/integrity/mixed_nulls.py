# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains Mixed Nulls check."""
from collections import defaultdict
from typing import Union, Dict, List, Iterable

import numpy as np
import pandas as pd

from deepchecks import Dataset, CheckResult, ensure_dataframe_type
from deepchecks.base.check import SingleDatasetBaseCheck, ConditionResult
from deepchecks.utils.dataframes import filter_columns_with_validation
from deepchecks.utils.features import calculate_feature_importance_or_null, column_importance_sorter_df
from deepchecks.utils.strings import string_baseform, format_percent, format_columns_for_condition
from deepchecks.utils.typing import Hashable
from deepchecks.errors import DeepchecksValueError


__all__ = ['MixedNulls']


DEFAULT_NULL_VALUES = {'none', 'null', 'nan', 'na', '', '\x00', '\x00\x00'}


class MixedNulls(SingleDatasetBaseCheck):
    """Search for various types of null values in a string column(s), including string representations of null.

    Args:
        null_string_list (List[str]):
            List of strings to be considered alternative null representations
        check_nan(bool):
            Whether to add to null list to check also NaN values
        columns (Union[Hashable, List[Hashable]]):
            Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[Hashable, List[Hashable]]):
            Columns to ignore, if none given checks based on columns variable
        n_top_columns (int): (optinal - used only if model was specified)
          amount of columns to show ordered by feature importance (date, index, label are first)
    """

    def __init__(
        self,
        null_string_list: Iterable[str] = None,
        check_nan: bool = True,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        n_top_columns: int = 10
    ):
        super().__init__()
        self.null_string_list = null_string_list
        self.check_nan = check_nan
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_columns = n_top_columns

    def run(self, dataset, model = None) -> CheckResult:
        """Run check.

        Args:
            dataset (Dataset):

        Returns:
            (CheckResult): DataFrame with columns ('Column Name', 'Value', 'Count', 'Percentage') for any column which
            have more than 1 null values.
        """
        feature_importances = calculate_feature_importance_or_null(dataset, model)
        return self._mixed_nulls(dataset, feature_importances)

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

    def _mixed_nulls(self, dataset: Union[pd.DataFrame, Dataset], feature_importances: pd.Series = None) -> CheckResult:
        """Run check logic.

        Args:
            dataset (DataFrame): dataset to check

        Returns
            (CheckResult): DataFrame with columns ('Column Name', 'Value', 'Count', 'Fraction of data') for any column
            which have more than 1 null values.
        """
        # Validate parameters
        original_dataset = dataset
        dataset: pd.DataFrame = ensure_dataframe_type(dataset)
        dataset = filter_columns_with_validation(dataset, self.columns, self.ignore_columns)
        null_string_list: set = self._validate_null_string_list(self.null_string_list, self.check_nan)

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
                percent = count / len(column_data)
                display_array.append([column_name, null_value, count, format_percent(percent)])
                result_dict[column_name][null_value] = {'count': count, 'percent': percent}

        # Create dataframe to display table
        if display_array:
            df_graph = pd.DataFrame(display_array, columns=['Column Name', 'Value', 'Count', 'Percent of data'])
            df_graph = df_graph.set_index(['Column Name', 'Value'])
            df_graph = column_importance_sorter_df(df_graph, original_dataset, feature_importances,
                                                   self.n_top_columns, col='Column Name')
            display = df_graph
        else:
            display = None

        return CheckResult(result_dict, display=display)

    def add_condition_different_nulls_not_more_than(self, max_allowed_null_types: int = 1):
        """Add condition - require column not to have more than given number of different null values.

        Args:
            max_allowed_null_types (int): Number of different null value types which is the maximum allowed.
        """
        def condition(result: Dict) -> ConditionResult:
            not_passing_columns = []
            for column in result.keys():
                nulls = result[column]
                num_nulls = len(nulls)
                if num_nulls > max_allowed_null_types:
                    not_passing_columns.append(column)
            if not_passing_columns:
                not_passing_str = ', '.join(map(str, not_passing_columns))
                return ConditionResult(False,
                                       f'Found columns with more than {max_allowed_null_types} null types: '
                                       f'{not_passing_str}')
            else:
                return ConditionResult(True)

        column_names = format_columns_for_condition(self.columns, self.ignore_columns)
        return self.add_condition(f'Not more than {max_allowed_null_types} different null types for {column_names}',
                                  condition)
