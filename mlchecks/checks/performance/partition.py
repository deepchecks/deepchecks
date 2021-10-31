from typing import List, Callable

import numpy as np
import pandas as pd
from mlchecks import Dataset


__all__ = ['partition_feature_to_bins', 'MLChecksFilter']

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
    # # add an epsilon to last edge because segment is [a,b)
    # if percentile_values[-1] in repeating_values:
    #     eps = np.finfo(np.float32).eps
    #     return_percentile_values.append(percentile_values[-1] + eps)
    # else:
    #     return_percentile_values.append(percentile_values[-1])

def partition_feature_to_bins(
    dataset: Dataset, column_name: str, max_segments: int, max_cat_proportions: float = 0.9
):
    """for categorical we'll have a max of max_segments+1, for the "Others".
    categories will contain at most max_cat_proportions% of the data, the rest will go to Other even if less then
    max_segments
    """
    column = dataset.data[column_name]
    if column_name not in dataset.cat_features():
        percentile_values = numeric_segmentation_edges(column, max_segments)
        filters = []
        for start, end in zip(percentile_values[:-1], percentile_values[1:]):
            # In case of the last edge, the end is closed.
            if end == percentile_values[-1]:
                f = lambda df: (df[column_name] >= start) & (df[column_name] <= end)
                label = f'[{format_number(start)} - {format_number(end)}]'
            else:
                f = lambda df: (df[column_name] >= start) & (df[column_name] < end)
                label = f'[{format_number(start)} - {format_number(end)})'

            filters.append(MLChecksFilter(f, label))
        return filters
    else:
        cat_hist_dict = column.value_counts().to_dict()
        sorted_hist = {k: v for k, v in sorted(cat_hist_dict.items(), key=lambda item: item[1], reverse=True)}
        argsorted_keys = np.array(list(sorted_hist.keys()))
        argsorted_keys = pd.Series(argsorted_keys).astype(column.dtypes).values

        first_less_then_max_cat_proportions_idx = np.argwhere(
            np.array(list(sorted_hist.values())).cumsum() > len(column) * max_cat_proportions
        )
        if first_less_then_max_cat_proportions_idx.shape[0] > 0:  # if there is such a category
            first_less_then_max_cat_proportions_idx = first_less_then_max_cat_proportions_idx[0][0]
        else:
            first_less_then_max_cat_proportions_idx = max_segments

        n_large_cats = min(max_segments, len(argsorted_keys), first_less_then_max_cat_proportions_idx + 1)

        filters = []
        for i in range(n_large_cats):
            f = lambda df: df[column_name] == argsorted_keys[i]
            filters.append(MLChecksFilter(f, argsorted_keys[i]))

        if len(argsorted_keys) > n_large_cats:
            f = lambda df: ~df[column_name].isin(argsorted_keys[:n_large_cats])
            filters.append(MLChecksFilter(f, 'Others'))

        return filters
