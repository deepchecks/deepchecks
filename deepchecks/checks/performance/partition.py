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
"""Module of functions to partition columns into segments."""
from copy import deepcopy
from typing import List, Callable

import numpy as np
import pandas as pd
from deepchecks import Dataset
from deepchecks.utils.strings import format_number
from deepchecks.utils.typing import Hashable


__all__ = ['partition_column', 'DeepchecksFilter']


class DeepchecksFilter:
    """Contains a filter function which works on a dataframe and a label describing the filter.

    Args:
        filter_func (Callable): function which receive dataframe and return a filter on it
        label (str): name of the filter
    """

    def __init__(self, filter_func: Callable, label: str):
        self.filter_func = filter_func
        self.label = label

    def filter(self, dataframe):
        """Run the filter on given dataframe."""
        return dataframe.loc[self.filter_func(dataframe)]


def numeric_segmentation_edges(column: pd.Series, max_segments: int) -> List[DeepchecksFilter]:
    """Split given series into values which are used to create quantiles segments.

    Tries to create `max_segments + 1` values (since segment is a range, so 2 values needed to create segment) but in
    case some quantiles have the same value they will be filtered, and the result will have less than max_segments + 1
    values.
    """
    percentile_values = np.array([min(column), max(column)])
    attempt_max_segments = max_segments
    prev_percentile_values = deepcopy(percentile_values)
    while len(percentile_values) < max_segments + 1:
        prev_percentile_values = deepcopy(percentile_values)
        percentile_values = pd.unique(
            np.nanpercentile(column.to_numpy(), np.linspace(0, 100, attempt_max_segments + 1))
        )
        attempt_max_segments *= 2

    if len(percentile_values) > max_segments + 1:
        percentile_values = prev_percentile_values

    return percentile_values


def largest_category_index_up_to_ratio(histogram, max_segments, max_cat_proportions):
    """Decide which categorical values are big enough to display individually.

    First check how many of the biggest categories needed in order to occupy `max_cat_proportions`% of the data. If
    the number is less than max_segments than return it, else return max_segments or number of unique values.
    """
    total_values = sum(histogram.values)
    first_less_then_max_cat_proportions_idx = np.argwhere(
        histogram.values.cumsum() >= total_values * max_cat_proportions
    )[0][0]

    # Get index of last value in histogram to show
    return min(max_segments, histogram.size, first_less_then_max_cat_proportions_idx + 1)


def partition_column(
    dataset: Dataset,
    column_name: Hashable,
    max_segments: int,
    max_cat_proportions: float = 0.9
) -> List[DeepchecksFilter]:
    """Split column into segments.

    For categorical we'll have a max of max_segments + 1, for the 'Others'. We take the largest categories which
    cumulative percent in data is equal/larger than `max_cat_proportions`. the rest will go to 'Others' even if less
    than max_segments.
    For numerical we split into maximum number of `max_segments` quantiles. if some of the quantiles are duplicates
    then we merge them into the same segment range (so not all ranges necessarily will have same size).

    Args:
        dataset (Dataset):
        column_name (Hashable): column to partition.
        max_segments (int): maximum number of segments to split into.
        max_cat_proportions (float): (for categorical) ratio to aggregate largest values to show.
    """
    column = dataset.data[column_name]
    if column_name not in dataset.cat_features:
        percentile_values = numeric_segmentation_edges(column, max_segments)
        # If for some reason only single value in the column (and column not categorical) we will get single item
        if len(percentile_values) == 1:
            f = lambda df, val=percentile_values[0]: (df[column_name] == val)
            label = str(percentile_values[0])
            return [DeepchecksFilter(f, label)]

        filters = []
        for start, end in zip(percentile_values[:-1], percentile_values[1:]):
            # In case of the last range, the end is closed.
            if end == percentile_values[-1]:
                f = lambda df, a=start, b=end: (df[column_name] >= a) & (df[column_name] <= b)
                label = f'[{format_number(start)} - {format_number(end)}]'
            else:
                f = lambda df, a=start, b=end: (df[column_name] >= a) & (df[column_name] < b)
                label = f'[{format_number(start)} - {format_number(end)})'

            filters.append(DeepchecksFilter(f, label))
        return filters
    else:
        # Get sorted histogram
        cat_hist_dict = column.value_counts()
        # Get index of last value in histogram to show
        n_large_cats = largest_category_index_up_to_ratio(cat_hist_dict, max_segments, max_cat_proportions)

        filters = []
        for i in range(n_large_cats):
            f = lambda df, val = cat_hist_dict.index[i]: df[column_name] == val
            filters.append(DeepchecksFilter(f, str(cat_hist_dict.index[i])))

        if len(cat_hist_dict) > n_large_cats:
            f = lambda df, values=cat_hist_dict.index[:n_large_cats]: ~df[column_name].isin(values)
            filters.append(DeepchecksFilter(f, 'Others'))

        return filters
