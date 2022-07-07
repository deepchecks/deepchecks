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
"""String length outlier check."""
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import stats

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular.utils.messages import get_condition_passed_message
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.features import N_TOP_MESSAGE, column_importance_sorter_df
from deepchecks.utils.strings import format_number, format_percent, is_string_column
from deepchecks.utils.typing import Hashable

__all__ = ['StringLengthOutOfBounds']


class StringLengthOutOfBounds(SingleDatasetCheck):
    """Detect strings with length that is much longer/shorter than the identified "normal" string lengths.

    Parameters
    ----------
    columns :Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable
    num_percentiles : int , default: 1000
        Number of percentiles values to retrieve for the length of the samples in the string
        column. Affects the resolution of string lengths that is used to detect outliers.
    inner_quantile_range : int , default: 94
        The int upper percentile [0-100] defining the inner percentile range.
        E.g. for 98 the range would be 2%-98%.
    outlier_factor : int , default: 4
        Strings would be defined as outliers if their length is outlier_factor times more/less
        than the values inside the inner quantile range.
    min_length_difference : int , default: 5
        The minimum length difference to be considered as outlier.
    min_length_ratio_difference : int , default: 0.5
        Used to calculate the minimum length difference to be considered as outlier. (calculated form this times the
        average of the normal lengths.)
    min_unique_value_ratio : float , default: 0.01
        Min
    min_unique_values : int , default: 100
        Minimum unique values in column to calculate string length outlier
    n_top_columns : int , optional
        amount of columns to show ordered by feature importance (date, index, label are first)
    outlier_length_to_show :int , default: 50
        Maximum length of outlier to show in results. If an outlier is longer it is trimmed and added '...'
    samples_per_range_to_show : int , default: 3
        Number of outlier samples to show in results per outlier range found.
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable]] = None,
        ignore_columns: Union[Hashable, List[Hashable]] = None,
        num_percentiles: int = 1000,
        inner_quantile_range: int = 94,
        outlier_factor: int = 4,
        min_length_difference: int = 5,
        min_length_ratio_difference: float = 0.5,
        min_unique_value_ratio: float = 0.01,
        min_unique_values: int = 100,
        n_top_columns: int = 10,
        outlier_length_to_show: int = 50,
        samples_per_range_to_show: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.num_percentiles = num_percentiles
        self.inner_quantile_range = inner_quantile_range
        self.outlier_factor = outlier_factor
        self.n_top_columns = n_top_columns
        self.min_length_difference = min_length_difference
        self.min_length_ratio_difference = min_length_ratio_difference
        self.min_unique_value_ratio = min_unique_value_ratio
        self.min_unique_values = min_unique_values
        self.outlier_length_to_show = outlier_length_to_show
        self.samples_per_range_to_show = samples_per_range_to_show

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind)
        df = select_from_dataframe(dataset.data, self.columns, self.ignore_columns)

        display_format = []
        results = {}

        for column_name in df.columns:
            column: Series = df[column_name].dropna()
            if column_name in dataset.cat_features or not is_string_column(column):
                continue

            results[column_name] = {'outliers': []}
            string_length_column = column.map(lambda x: len(str(x)))

            # If not a lot of unique values, calculate the percentiles for existing values.
            if string_length_column.nunique() < self.num_percentiles:
                string_length_column = string_length_column.sort_values()
                quantile_list = 100 * stats.rankdata(string_length_column, 'ordinal') / len(string_length_column)
                percentile_histogram = {quantile_list[i]: string_length_column.iloc[i] for i in
                                        range(len(string_length_column))}
            else:
                quantile_list = list(np.linspace(0.0, 100.0, self.num_percentiles + 1))
                quantile_values = np.percentile(string_length_column, quantile_list, interpolation='nearest')
                percentile_histogram = dict(zip(quantile_list, list(quantile_values)))

            outlier_sections = outlier_on_percentile_histogram(percentile_histogram, self.inner_quantile_range,
                                                               self.outlier_factor)
            if outlier_sections:
                quantiles_not_in_section = \
                    [x for x in quantile_list if all((not _in_range(x, a, b)) for a, b in outlier_sections)]
                non_outlier_section = (min(quantiles_not_in_section), max(quantiles_not_in_section))

                non_outlier_lower_limit = percentile_histogram[non_outlier_section[0]]
                non_outlier_upper_limit = percentile_histogram[non_outlier_section[1]]

                # add to result
                for outlier_section in outlier_sections:
                    lower_range, upper_range = self._filter_outlier_section(percentile_histogram[outlier_section[0]],
                                                                            percentile_histogram[outlier_section[1]],
                                                                            non_outlier_lower_limit,
                                                                            non_outlier_upper_limit)
                    if lower_range > upper_range:
                        continue

                    outlier_samples = string_length_column[
                        string_length_column.between(lower_range, upper_range, inclusive='both')]

                    if not outlier_samples.empty:
                        outlier_examples = column[outlier_samples[:self.samples_per_range_to_show].index]
                        outlier_examples = [trim(x, self.outlier_length_to_show) for x in outlier_examples]

                        results[column_name]['normal_range'] = {
                            'min': non_outlier_lower_limit,
                            'max': non_outlier_upper_limit
                        }
                        results[column_name]['n_samples'] = column.size
                        results[column_name]['outliers'].append({
                            'range': {'min': lower_range,
                                      'max': upper_range
                                      },
                            'n_samples': outlier_samples.size
                        })

                        if context.with_display:
                            display_format.append([column_name,
                                                   f'{format_number(non_outlier_lower_limit)} -'
                                                   f' {format_number(non_outlier_upper_limit)}',
                                                   f'{format_number(lower_range)} -'
                                                   f' {format_number(upper_range)}',
                                                   f'{outlier_samples.size}',
                                                   outlier_examples
                                                   ])

        # Create dataframe to display graph
        if display_format:
            df_graph = DataFrame(display_format,
                                 columns=['Column Name',
                                          'Range of Detected Normal String Lengths',
                                          'Range of Detected Outlier String Lengths',
                                          'Number of Outlier Samples',
                                          'Example Samples'])
            df_graph = df_graph.set_index(['Column Name',
                                           'Range of Detected Normal String Lengths',
                                           'Range of Detected Outlier String Lengths'])

            df_graph = column_importance_sorter_df(df_graph, dataset, context.feature_importance,
                                                   self.n_top_columns, col='Column Name')
            display = [N_TOP_MESSAGE % self.n_top_columns, df_graph]
        else:
            display = None

        return CheckResult(results, display=display)

    def _filter_outlier_section(self, lower_range, upper_range, non_outlier_lower_range, non_outlier_upper_range):
        lower_range_distance = lower_range - non_outlier_upper_range
        higher_range_distance = non_outlier_lower_range - upper_range

        non_outlier_range_average = (non_outlier_upper_range + non_outlier_lower_range) / 2

        minimum_difference = max(self.min_length_difference,
                                 self.min_length_ratio_difference * non_outlier_range_average)
        if lower_range_distance > 0:
            if lower_range_distance < minimum_difference:
                lower_range += minimum_difference - lower_range_distance
        elif higher_range_distance > 0:
            if higher_range_distance < minimum_difference:
                upper_range -= minimum_difference - higher_range_distance

        return lower_range, upper_range

    def add_condition_number_of_outliers_less_or_equal(self, max_outliers: int = 0):
        """Add condition - require column's number of string length outliers to be less or equal to the threshold.

        Parameters
        ----------
        max_outliers : int , default: 0
            Number of string length outliers which is the maximum allowed.
        """
        def compare_outlier_count(result: Dict) -> ConditionResult:
            not_passing_columns = {}
            for column_name in result.keys():
                column = result[column_name]
                total_outliers = sum((outlier['n_samples'] for outlier in column['outliers']))
                if total_outliers > max_outliers:
                    not_passing_columns[column_name] = total_outliers
            if not_passing_columns:
                details = f'Found {len(not_passing_columns)} out of {len(result)} columns with number of outliers ' \
                          f'above threshold: {not_passing_columns}'
                return ConditionResult(ConditionCategory.FAIL, details)
            else:
                details = f'Passed for {len(result)} columns'
                return ConditionResult(ConditionCategory.PASS, details)

        return self.add_condition(
            f'Number of string length outliers is less or equal to {max_outliers}',
            compare_outlier_count)

    def add_condition_ratio_of_outliers_less_or_equal(self, max_ratio: float = 0):
        """Add condition - require column's ratio of string length outliers to be less or equal to threshold.

        Parameters
        ----------
        max_ratio : float , default: 0
            Maximum allowed string length outliers ratio.
        """
        def compare_outlier_ratio(result: Dict):
            not_passing_columns = {}
            for column_name in result.keys():
                column = result[column_name]
                total_outliers = sum((outlier['n_samples'] for outlier in column['outliers']))
                ratio = total_outliers / column['n_samples'] if total_outliers > 0 else 0
                if ratio > max_ratio:
                    not_passing_columns[column_name] = format_percent(ratio)
            if not_passing_columns:
                details = f'Found {len(not_passing_columns)} out of {len(result)} relevant columns with outliers ' \
                          f'ratio above threshold: {not_passing_columns}'
                return ConditionResult(ConditionCategory.WARN, details)
            else:
                return ConditionResult(ConditionCategory.PASS, get_condition_passed_message(result))

        return self.add_condition(
            f'Ratio of string length outliers is less or equal to {format_percent(max_ratio)}',
            compare_outlier_ratio)


def outlier_on_percentile_histogram(percentile_histogram: Dict[float, float], iqr_percent: float = 85,
                                    outlier_factor: float = 5) -> Tuple[Tuple[float, float]]:
    """Get outlier ranges on histogram.

    Parameters
    ----------
    percentile_histogram : Dict[float, float]
        histogram to search for outliers in shape [0.0-100.0]->[float]
    iqr_percent : float , default: 85
        Interquartile range upper percentage, start searching for outliers outside IQR.
    outlier_factor : float , default: 5
        a factor to consider outlier.
    Returns
    -------
    Tuple[Tuple[float, float]]
        percent ranges in the histogram that are outliers, empty tuple if none is found
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


def _in_range(x, a, b):
    return a <= x <= b


def trim(x, max_length):
    if len(x) <= max_length:
        return x
    return x[:max_length] + '...'
