"""String length outlier check."""
from functools import reduce
from typing import Union, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import stats

from mlchecks import CheckResult, SingleDatasetBaseCheck, Dataset, ensure_dataframe_type
from mlchecks.string_utils import is_string_column, format_number
from mlchecks.base.dataframe_utils import filter_columns_with_validation

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
    """Detect strings with length that is much longer/shorter than the identified "normal" string lengths."""

    def __init__(self, columns: Union[str, Iterable[str]] = None, ignore_columns: Union[str, Iterable[str]] = None,
                 num_percentiles: int = 1000, inner_quantile_range: int = 94, outlier_factor: int = 4):
        """Initialize the StringLengthOutOfBounds check.

        Args:
            columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored
                        ones.
            ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns
                            variable
            num_percentiles (int): Number of percentiles values to retrieve for the length of the samples in the string
                                   column. Affects the resolution of string lengths that is used to detect outliers.
            inner_quantile_range(int): The int upper percentile [0-100] defining the inner percentile range.
                                       E.g. for 98 the range would be 2%-98%.
            outlier_factor (int): Strings would be defined as outliers if their length is outlier_factor times more/less
                                  than the values inside the inner quantile range.
        """
        super().__init__()
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.num_percentiles = num_percentiles
        self.inner_quantile_range = inner_quantile_range
        self.outlier_factor = outlier_factor

    def run(self, dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            dataset (DataFrame): A dataset or pd.FataFrame object.
        """
        return self._string_length_out_of_bounds(dataset)

    def _string_length_out_of_bounds(self, dataset: Union[pd.DataFrame, Dataset]) -> CheckResult:
        # Validate parameters
        df: pd.DataFrame = ensure_dataframe_type(dataset)
        df = filter_columns_with_validation(df, self.columns, self.ignore_columns)

        results = []

        for column_name in df.columns:
            column: Series = df[column_name]

            if not is_string_column(column):
                continue

            string_length_column = column.map(len)

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
                        results.append([column_name,
                                        f'{format_number(percentile_histogram[non_outlier_section[0]])} -'
                                        f' {format_number(percentile_histogram[non_outlier_section[1]])}',
                                        f'{format_number(percentile_histogram[outlier_section[0]])} -'
                                        f' {format_number(percentile_histogram[outlier_section[1]])}',
                                        f'{n_outlier_samples}'
                                        ])

        # Create dataframe to display graph
        df_graph = DataFrame(results,
                             columns=['Column Name',
                                      'Range of Detected Normal String Lengths',
                                      'Range of Detected Outlier String Lengths',
                                      'Number of Outlier Samples'])
        df_graph = df_graph.set_index(['Column Name',
                                       'Range of Detected Normal String Lengths',
                                       'Range of Detected Outlier String Lengths'])

        display = df_graph if len(df_graph) > 0 else None

        return CheckResult(df_graph, check=self.__class__, display=display)
