"""Module of functions to partition columns into segments."""
from typing import List, Callable

import numpy as np
import pandas as pd
from mlchecks import Dataset


__all__ = ['partition_column', 'MLChecksFilter']

from mlchecks.string_utils import format_number


class MLChecksFilter:
    """Use this class to mark to not filter by equal but filter by not equal values."""

    def __init__(self, filter_func: Callable, label: str):
        """Init MLChecksFilter.

        Args:
            filter_func (Callable): function which receive dataframe and return a filter on it
            label (str): name of the filter
        """
        self.filter_func = filter_func
        self.label = label

    def filter(self, dataframe):
        """Run the filter on given dataframe."""
        return dataframe.loc[self.filter_func(dataframe)]


def numeric_segmentation_edges(column: pd.Series, max_segments: int) -> List[MLChecksFilter]:
    percentile_values = np.nanpercentile(column.to_numpy(), np.linspace(0, 100, max_segments + 1))
    # If there are a lot of duplicate values, some of the quantiles might be equal,
    # so filter them (with preserved order)
    return pd.unique(percentile_values)


def largest_category_index_up_to_ratio(histogram, max_segments, max_cat_proportions):
    total_values = sum(histogram.values)
    first_less_then_max_cat_proportions_idx = np.argwhere(
        histogram.values.cumsum() > total_values * max_cat_proportions
    )
    if first_less_then_max_cat_proportions_idx.shape[0] > 0:  # if there is such a category
        first_less_then_max_cat_proportions_idx = first_less_then_max_cat_proportions_idx[0][0]
    else:
        first_less_then_max_cat_proportions_idx = max_segments

    # Get index of last value in histogram to show
    return min(max_segments, histogram.size, first_less_then_max_cat_proportions_idx + 1)


def partition_column(dataset: Dataset, column_name: str, max_segments: int, max_cat_proportions: float = 0.9)\
        -> List[MLChecksFilter]:
    """Split column into segments.

    For categorical we'll have a max of max_segments + 1, for the 'Others'. We take the largest categories which
    cumulative percent in data is equal/larger than `max_cat_proportions`. the rest will go to 'Others' even if less
    than max_segments.
    For numerical we split into `max_segments` number of quantiles.

    Args:
        dataset (Dataset):
        column_name (str):
        max_segments (int): maximum number of segments to split into.
        max_cat_proportions (float): (for categorical) ratio to aggregate largest values to show.
    """
    column = dataset.data[column_name]
    if column_name not in dataset.cat_features():
        percentile_values = numeric_segmentation_edges(column, max_segments)
        filters = []
        for start, end in zip(percentile_values[:-1], percentile_values[1:]):
            # In case of the last edge, the end is closed.
            if end == percentile_values[-1]:
                f = lambda df, a=start, b=end: (df[column_name] >= a) & (df[column_name] <= b)
                label = f'[{format_number(start)} - {format_number(end)}]'
            else:
                f = lambda df, a=start, b=end: (df[column_name] >= a) & (df[column_name] < b)
                label = f'[{format_number(start)} - {format_number(end)})'

            filters.append(MLChecksFilter(f, label))
        return filters
    else:
        # Get sorted histogram
        cat_hist_dict = column.value_counts()
        # Get index of last value in histogram to show
        n_large_cats = largest_category_index_up_to_ratio(cat_hist_dict, max_segments, max_cat_proportions)

        filters = []
        for i in range(n_large_cats):
            f = lambda df, val = cat_hist_dict.index[i]: df[column_name] == val
            filters.append(MLChecksFilter(f, cat_hist_dict.index[i]))

        if len(cat_hist_dict) > n_large_cats:
            f = lambda df, values=cat_hist_dict.index[:n_large_cats]: ~df[column_name].isin(values)
            filters.append(MLChecksFilter(f, 'Others'))

        return filters
