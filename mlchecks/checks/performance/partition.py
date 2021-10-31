from typing import List, Callable

import numpy as np
import pandas as pd
from mlchecks import Dataset


__all__ = ['partition_column', 'MLChecksFilter']

from mlchecks.string_utils import format_number


class MLChecksFilter:
    """Use this class to mark to not filter by equal but filter by not equal values."""

    def __init__(self, filter_func: Callable, label: str):
        self.filter_func = filter_func
        self.label = label

    def filter(self, dataframe):
        return dataframe.loc[self.filter_func(dataframe)]


def numeric_segmentation_edges(column: pd.Series, max_segments: int) -> List[MLChecksFilter]:
    percentile_values: List = (np.nanpercentile(column.astype(float).values, np.linspace(0, 100, max_segments + 1))
                               .tolist())

    # if have [a,a) segments, make it [a,a+eps)
    repeating_values = []
    return_percentile_values = []
    for i in range(len(percentile_values)):
        if percentile_values[i] not in repeating_values: # Matan: Why?
            return_percentile_values.append(percentile_values[i])
            repeating_values.append(percentile_values[i])

    return return_percentile_values


def partition_column(dataset: Dataset, column_name: str, max_segments: int, max_cat_proportions: float = 0.9)\
        -> List[MLChecksFilter]:
    """for categorical we'll have a max of max_segments+1, for the 'Others'.
    categories will contain at most max_cat_proportions% of the data, the rest will go to 'Others' even if less then
    max_segments.

    Args:
        dataset (Dataset):
        column_name (str):
        max_segments (int):
        max_cat_proportions (float):
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
        # Find first value which accumulated sum is larger than max proportion
        first_less_then_max_cat_proportions_idx = np.argwhere(
            cat_hist_dict.values.cumsum() > len(column) * max_cat_proportions
        )
        if first_less_then_max_cat_proportions_idx.shape[0] > 0:  # if there is such a category
            first_less_then_max_cat_proportions_idx = first_less_then_max_cat_proportions_idx[0][0]
        else:
            first_less_then_max_cat_proportions_idx = max_segments

        # Get index of last value in histogram to show
        n_large_cats = min(max_segments, len(cat_hist_dict), first_less_then_max_cat_proportions_idx + 1)

        filters = []
        for i in range(n_large_cats):
            f = lambda df, val = cat_hist_dict.index[i]: df[column_name] == val
            filters.append(MLChecksFilter(f, cat_hist_dict.index[i]))

        if len(cat_hist_dict) > n_large_cats:
            f = lambda df, values=cat_hist_dict.index[n_large_cats:]: ~df[column_name].isin(values)
            filters.append(MLChecksFilter(f, 'Others'))

        return filters
