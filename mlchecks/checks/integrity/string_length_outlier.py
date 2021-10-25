"""String mismatch functions."""
from functools import reduce
from typing import Union, Dict, Iterable, Tuple
from math import ceil, floor

import numpy as np
import pandas as pd
from pandas import DataFrame, StringDtype, Series

from mlchecks import CheckResult, SingleDatasetBaseCheck, Dataset, ensure_dataframe_type
from mlchecks.base import string_utils
from mlchecks.base.dataframe_utils import filter_columns_with_validation

__all__ = ['string_length_outlier', 'StringLengthOutlier']

from mlchecks.utils import MLChecksValueError


def string_length_outlier(dataset: Union[pd.DataFrame, Dataset], columns: Union[str, Iterable[str]] = None,
                          ignore_columns: Union[str, Iterable[str]] = None,
                          num_percentiles: int = 1000, inner_quantile_range: int = 94,
                          outlier_factor: int = 4) -> CheckResult:
    """Detect outliers in a categorical column[s].

    Args:
        dataset (DataFrame): A dataset or pd.FataFrame object.
        columns (Union[str, Iterable[str]]): Columns to check, if none are given checks all columns except ignored ones.
        ignore_columns (Union[str, Iterable[str]]): Columns to ignore, if none given checks based on columns variable
        num_percentiles (int): Number of percentiles values to retrieve for the length of the samples in the string
                               column. Affects the resolution of string lengths that is used to detect outliers.
        inner_quantile_range(int): The int upper percentile [0-100] defining the inner percentile range.
                                   E.g. for 98 the range would be 2%-98%.
        outlier_factor (int): Strings would be defined as outliers if their length is outlier_factor times more/less
                              than the values inside the inner quantile range.
    """
    # Validate parameters
    df: pd.DataFrame = ensure_dataframe_type(dataset)
    df = filter_columns_with_validation(df, columns, ignore_columns)

    results = []

    for column_name in df.columns:
        column: Series = df[column_name]

        if not string_utils.is_string_column(column):
            continue

        string_length_column = column.map(len)

        quantile_list = list(np.linspace(0.0, 100.0, num_percentiles + 1))
        quantile_values = np.percentile(string_length_column, quantile_list)

        percentile_histogram = dict(zip(quantile_list, list(quantile_values)))

        outlier_sections = outlier_on_percentile_histogram(percentile_histogram, inner_quantile_range,
                                                           outlier_factor)
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
                                    f'{ceil(percentile_histogram[non_outlier_section[0]])} -'
                                    f' {floor(percentile_histogram[non_outlier_section[1]])}',
                                    f'{ceil(percentile_histogram[outlier_section[0]])} -'
                                    f' {floor(percentile_histogram[outlier_section[1]])}',
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

    return CheckResult(df_graph, check=string_length_outlier, display=display)


def in_range(x, a, b):
    return a <= x <= b


def outlier_on_percentile_histogram(percentile_histogram: Dict[float, float], iqr_percent: float = 85,
                                    outlier_factor: float = 5) -> Tuple[Tuple[float, float]]:
    """Get outlier ranges on histogram.

    Args:
        percentile_histogram (Dict[float, float]):
        iqr_percent (float): Interquartile range percentage, start searching from
        outlier_factor (float): a factor to consider outlier

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


class StringLengthOutlier(SingleDatasetBaseCheck):
    """Search for string length outlier in categorical column[s]."""

    def run(self, dataset, model=None) -> CheckResult:
        """Run string_length_outlier check.

        Args:
            dataset (DataFrame): A dataset or pd.FataFrame object.
        """
        return string_length_outlier(dataset, **self.params)
