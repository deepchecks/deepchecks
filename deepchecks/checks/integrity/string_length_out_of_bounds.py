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
"""String length outlier check."""
from collections import defaultdict
from functools import reduce
from typing import Union, Dict, Tuple, List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import stats

from deepchecks import CheckResult, SingleDatasetBaseCheck, Dataset, ensure_dataframe_type, ConditionResult
from deepchecks.utils.features import calculate_feature_importance_or_null, column_importance_sorter_df
from deepchecks.utils.strings import is_string_column, format_number, format_columns_for_condition, format_percent
from deepchecks.utils.dataframes import filter_columns_with_validation
from deepchecks.utils.typing import Hashable


__all__ = ['StringLengthOutOfBounds']


def in_range(x, a, b):
    return a <= x <= b


def outlier_on_percentile_histogram(percentile_histogram: Dict[float, float], iqr_percent: float = 85,
                                    outlier_factor: float = 5) -> Tuple[Tuple[float, float]]:
    """Get outlier ranges on histogram.

    Args:
        percentile_histogram (Dict[float, float]): histogram to search for outliers in shape [0.0-100.0]->[float]
        iqr_percent (float): Interquartile range upper percentage, start searching for outliers outside IQR.
        outlier_factor (float): a factor to consider outlier.

    Returns:
         (Tuple[Tuple[float, float]]): percent ranges in the histogram that are outliers, empty tuple if none is found
    """
    if any((k < 0) or k > 100 for k in percentile_histogram.keys()):
        raise ValueError('dict keys must be percentiles between 0 and 100')
    if any((v < 0) for v in percentile_histogram.values()):
        raise ValueError('dict values must be counts that are non-negative numbers')

    percentile_df = pd.DataFrame.from_dict(percentile_histogram, orient='index')

    # calculate IQR with iqr_percent
    closest_point_upper = np.argmin(np.abs(iqr_percent - percentile_df.index.values))
    closest_point_lower = np.argmin(np.abs(100 - iqr_percent - percentile_df.index.values))
    center_point = np.argmin(np.abs(50 - percentile_df.index.values))

    iqr = np.abs(percentile_df.iloc[closest_point_upper] - percentile_df.iloc[closest_point_lower])

    outlier_df = percentile_df[
        (np.abs(percentile_df - percentile_df.iloc[center_point])
         > outlier_factor * iqr / 2).values
    ]

    outlier_section_list = []
    lower_outlier_range = outlier_df[outlier_df.index < 50]
    if lower_outlier_range.shape[0] > 0:
        outlier_section_list.append((lower_outlier_range.index.values[0], lower_outlier_range.index.values[-1]))

    upper_outlier_range = outlier_df[outlier_df.index > 50]
    if upper_outlier_range.shape[0] > 0:
        outlier_section_list.append((upper_outlier_range.index.values[0], upper_outlier_range.index.values[-1]))

    return tuple(outlier_section_list)


class StringLengthOutOfBounds(SingleDatasetBaseCheck):
    """Detect strings with length that is much longer/shorter than the identified "normal" string lengths.

    Args:
        columns (Union[Hashable, List[Hashable]]):
            Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[Hashable, List[Hashable]]):
            Columns to ignore, if none given checks based on columns variable
        num_percentiles (int):
            Number of percentiles values to retrieve for the length of the samples in the string
            column. Affects the resolution of string lengths that is used to detect outliers.
        inner_quantile_range(int):
            The int upper percentile [0-100] defining the inner percentile range.
            E.g. for 98 the range would be 2%-98%.
        outlier_factor (int):
            Strings would be defined as outliers if their length is outlier_factor times more/less
            than the values inside the inner quantile range.
        n_top_columns (int): (optinal - used only if model was specified)
          amount of columns to show ordered by feature importance (date, index, label are first)
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable]] = None,
        ignore_columns: Union[Hashable, List[Hashable]] = None,
        num_percentiles: int = 1000,
        inner_quantile_range: int = 94,
        outlier_factor: int = 4,
        n_top_columns: int = 10
    ):
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.num_percentiles = num_percentiles
        self.inner_quantile_range = inner_quantile_range
        self.outlier_factor = outlier_factor
        self.n_top_columns = n_top_columns

    def run(self, dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            dataset (DataFrame): A dataset or pd.FataFrame object.
        """
        feature_importances = calculate_feature_importance_or_null(dataset, model)
        return self._string_length_out_of_bounds(dataset, feature_importances)

    def _string_length_out_of_bounds(self, dataset: Union[pd.DataFrame, Dataset],
                                     feature_importances: pd.Series = None) -> CheckResult:
        # Validate parameters
        df: pd.DataFrame = ensure_dataframe_type(dataset)
        df = filter_columns_with_validation(df, self.columns, self.ignore_columns)

        display_format = []
        results = defaultdict(lambda: {'outliers': []})

        for column_name in df.columns:
            column: Series = df[column_name].dropna()

            if not is_string_column(column):
                continue

            string_length_column = column.map(lambda x: len(str(x)), na_action='ignore')

            # If not a lot of unique values, calculate the percentiles for existing values.
            if string_length_column.nunique() < self.num_percentiles:
                string_length_column = string_length_column.to_numpy()
                string_length_column.sort()
                quantile_list = 100 * stats.rankdata(string_length_column, 'ordinal') / len(string_length_column)
                percentile_histogram = {quantile_list[i]: string_length_column[i] for i in
                                        range(len(string_length_column))}
            else:
                quantile_list = list(np.linspace(0.0, 100.0, self.num_percentiles + 1))
                quantile_values = np.percentile(string_length_column, quantile_list, interpolation='nearest')
                percentile_histogram = dict(zip(quantile_list, list(quantile_values)))

            outlier_sections = outlier_on_percentile_histogram(percentile_histogram, self.inner_quantile_range,
                                                               self.outlier_factor)
            if outlier_sections:
                quantiles_not_in_section = \
                    [x for x in quantile_list if all((not in_range(x, a, b)) for a, b in outlier_sections)]
                non_outlier_section = (min(quantiles_not_in_section), max(quantiles_not_in_section))

                # add to result
                for outlier_section in outlier_sections:
                    n_outlier_samples = reduce(lambda value, x, ph=percentile_histogram, os=outlier_section:
                                               value + in_range(x, ph[os[0]], ph[os[1]]),
                                               string_length_column, 0)
                    if n_outlier_samples:
                        display_format.append([column_name,
                                               f'{format_number(percentile_histogram[non_outlier_section[0]])} -'
                                               f' {format_number(percentile_histogram[non_outlier_section[1]])}',
                                               f'{format_number(percentile_histogram[outlier_section[0]])} -'
                                               f' {format_number(percentile_histogram[outlier_section[1]])}',
                                               f'{n_outlier_samples}'
                                               ])
                        results[column_name]['normal_range'] = {
                                'min': percentile_histogram[non_outlier_section[0]],
                                'max': percentile_histogram[non_outlier_section[1]]
                            }
                        results[column_name]['n_samples'] = column.size
                        results[column_name]['outliers'].append({
                            'range': {'min': percentile_histogram[outlier_section[0]],
                                      'max': percentile_histogram[outlier_section[1]]
                                      },
                            'n_samples': n_outlier_samples
                        })

        # Create dataframe to display graph
        df_graph = DataFrame(display_format,
                             columns=['Column Name',
                                      'Range of Detected Normal String Lengths',
                                      'Range of Detected Outlier String Lengths',
                                      'Number of Outlier Samples'])
        df_graph = df_graph.set_index(['Column Name',
                                       'Range of Detected Normal String Lengths',
                                       'Range of Detected Outlier String Lengths'])

        df_graph = column_importance_sorter_df(df_graph, dataset, feature_importances,
                                               self.n_top_columns, col='Column Name')
        display = df_graph if len(df_graph) > 0 else None

        return CheckResult(results, display=display)

    def add_condition_number_of_outliers_not_greater_than(self, max_outliers: int = 0):
        """Add condition - require column not to have more than given number of string length outliers.

        Args:
            max_outliers (int): Number of string length outliers which is the maximum allowed.
        """
        def compare_outlier_count(result: Dict) -> ConditionResult:
            not_passing_columns = []
            for column_name in result.keys():
                column = result[column_name]
                total_outliers = 0
                for outlier in column['outliers']:
                    total_outliers += outlier['n_samples']

                if total_outliers > max_outliers:
                    not_passing_columns.append(column_name)
            if not_passing_columns:
                not_passing_str = ', '.join(map(str, not_passing_columns))
                return ConditionResult(False,
                                       f'Found columns with greater than {max_outliers} outliers: '
                                       f'{not_passing_str}')
            else:
                return ConditionResult(True)

        column_names = format_columns_for_condition(self.columns, self.ignore_columns)
        return self.add_condition(
            f'Number of outliers not greater than {max_outliers} string length outliers for {column_names}',
            compare_outlier_count)

    def add_condition_ratio_of_outliers_not_greater_than(self, max_ratio: float = 0):
        """Add condition - require column not to have more than given ratio of string length outliers.

        Args:
            max_ratio (int): Maximum allowed string length outliers ratio.
        """
        def compare_outlier_ratio(result: Dict):
            not_passing_columns = []
            for column_name in result.keys():
                column = result[column_name]
                total_outliers = 0
                for outlier in column['outliers']:
                    total_outliers += outlier['n_samples']

                if total_outliers/column['n_samples'] > max_ratio:
                    not_passing_columns.append(column_name)
            if not_passing_columns:
                not_passing_str = ', '.join(map(str, not_passing_columns))
                return ConditionResult(False,
                                       f'Found columns with greater than {format_percent(max_ratio)} outliers: '
                                       f'{not_passing_str}')
            else:
                return ConditionResult(True)

        column_names = format_columns_for_condition(self.columns, self.ignore_columns)
        return self.add_condition(
            f'Ratio of outliers not greater than {format_percent(max_ratio)} string length outliers for {column_names}',
            compare_outlier_ratio)
