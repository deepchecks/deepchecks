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
"""Module of functions to partition columns into segments."""
from collections import defaultdict
from copy import deepcopy
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.tree import _tree

from deepchecks.tabular.dataset import Dataset
from deepchecks.utils.strings import format_number
from deepchecks.utils.typing import Hashable

# TODO: move tabular functionality to the tabular sub-package


__all__ = ['partition_column', 'DeepchecksFilter', 'DeepchecksBaseFilter', 'convert_tree_leaves_into_filters',
           'intersect_two_filters', 'partition_numeric_feature_around_segment']


class DeepchecksFilter:
    """Contains a filter function which works on a dataframe and a label describing the filter.

    Parameters
    ----------
    filter_functions : List[Callable], default: None
        List of functions that receive a DataFrame and return a filter on it. If None, no filter is applied
    label : str, default = ''
        name of the filter
    """

    def __init__(self, filter_functions: List[Callable] = None, label: str = ''):
        if not filter_functions:
            self.filter_functions = []
        else:
            self.filter_functions = filter_functions
        self.label = label

    def filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Run the filter on given dataframe. Return rows in data frame satisfying the filter properties."""
        for func in self.filter_functions:
            dataframe = dataframe.loc[func(dataframe)]
        return dataframe


class DeepchecksBaseFilter(DeepchecksFilter):
    """Extend DeepchecksFilter class for feature range based filters.

    Parameters
    ----------
    filters: dict, default: None
        A dictionary in containing feature names as keys and the filtering range as value.
    filter_functions : List[Callable], default: None
        List of functions that receive a DataFrame and return a filter on it. If None, no filter is applied
    label : str, default = ''
        Name of the filter
    """

    def __init__(self, filters: dict = None, filter_functions: List[Callable] = None, label: str = ''):
        if filters is None:
            filters = defaultdict()
        self.filters = filters
        super().__init__(filter_functions, label)

    def add_filter(self, feature_name: str, threshold: float, greater_then: bool = True):
        """Add a filter by intersecting it with existing filter."""
        if greater_then:
            filter_func = [lambda df, a=threshold: df[feature_name] > a]
            if feature_name in self.filters.keys():
                original_range = self.filters[feature_name]
                self.filters[feature_name] = [max(threshold, original_range[0]), original_range[1]]
            else:
                self.filters[feature_name] = [threshold, np.inf]
        else:
            filter_func = [lambda df, a=threshold: df[feature_name] <= a]
            if feature_name in self.filters.keys():
                original_range = self.filters[feature_name]
                self.filters[feature_name] = [original_range[0], min(threshold, original_range[1])]
            else:
                self.filters[feature_name] = [np.NINF, threshold]
        self.filter_functions += filter_func
        return self

    def copy(self):
        """Return a copy of the object."""
        return DeepchecksBaseFilter(self.filters.copy(), self.filter_functions.copy(), self.label)


def intersect_two_filters(filter1: DeepchecksFilter, filter2: DeepchecksFilter) -> DeepchecksFilter:
    """Merge two DeepChecksFilters into one, an intersection of both filters."""
    return DeepchecksFilter(filter1.filter_functions + filter2.filter_functions)


def partition_numeric_feature_around_segment(column: pd.Series, segment: List[float],
                                             max_additional_segments: int = 4) -> np.ndarray:
    """Split given series into segments containing specified segment.

    Tries to create segments as balanced as possible in size.
    Parameters
    ----------
    column : pd.Series
        Series to be partitioned.
    segment : List[float]
        Segment to be included in the partition.
    max_additional_segments : int, default = 4
        Maximum number of segments to be returned (not including the original segment).
    """
    data_below_segment, data_above_segment = column[column <= segment[0]], column[column > segment[1]]
    if len(data_below_segment) + len(data_above_segment) == 0:
        return np.array([np.nanmin(column), np.nanmax(column)])
    ratio = np.divide(len(data_below_segment), len(data_below_segment) + len(data_above_segment))

    if len(data_below_segment) == 0:
        segments_below = np.array([np.nanmin(column)])
    elif data_below_segment.nunique() == 1:
        segments_below = np.array([np.nanmin(column), segment[0]])
    else:
        segments_below = numeric_segmentation_edges(data_below_segment, round(max_additional_segments * ratio))
        segments_below = np.append(np.delete(segments_below, len(segments_below) - 1), segment[0])

    if len(data_above_segment) == 0:
        segments_above = np.array([np.nanmax(column)])
    elif data_above_segment.nunique() == 1:
        segments_above = np.array([segment[1], np.nanmax(column)])
    else:
        segments_above = numeric_segmentation_edges(data_above_segment, round(max_additional_segments * (1 - ratio)))
        segments_above = np.append(segment[1], np.delete(segments_above, 0))

    return np.unique(np.concatenate([segments_below, segments_above], axis=None))


def numeric_segmentation_edges(column: pd.Series, max_segments: int) -> np.ndarray:
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
        if len(percentile_values) == len(prev_percentile_values):
            break
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
        max_segments: int = 10,
        max_cat_proportions: float = 0.9,
) -> List[DeepchecksFilter]:
    """Split column into segments.

    For categorical we'll have a max of max_segments + 1, for the 'Others'. We take the largest categories which
    cumulative percent in data is equal/larger than `max_cat_proportions`. the rest will go to 'Others' even if less
    than max_segments.
    For numerical we split into maximum number of `max_segments` quantiles. if some of the quantiles are duplicates
    then we merge them into the same segment range (so not all ranges necessarily will have same size).

    Parameters
    ----------
    dataset : Dataset
    column_name : Hashable
        column to partition.
    max_segments : int, default: 10
        maximum number of segments to split into.
    max_cat_proportions : float , default: 0.9
        (for categorical) ratio to aggregate largest values to show.

    Returns
    -------
    List[DeepchecksFilter]
    """
    column = dataset.data[column_name]
    if column_name in dataset.numerical_features:
        percentile_values = numeric_segmentation_edges(column, max_segments)
        # If for some reason only single value in the column (and column not categorical) we will get single item
        if len(percentile_values) == 1:
            f = lambda df, val=percentile_values[0]: (df[column_name] == val)
            label = str(percentile_values[0])
            return [DeepchecksFilter([f], label)]

        filters = []
        for start, end in zip(percentile_values[:-1], percentile_values[1:]):
            # In case of the last range, the end is closed.
            if end == percentile_values[-1]:
                f = lambda df, a=start, b=end: (df[column_name] >= a) & (df[column_name] <= b)
                label = f'[{format_number(start)} - {format_number(end)}]'
            else:
                f = lambda df, a=start, b=end: (df[column_name] >= a) & (df[column_name] < b)
                label = f'[{format_number(start)} - {format_number(end)})'

            filters.append(DeepchecksFilter([f], label))
        return filters
    elif column_name in dataset.cat_features:
        # Get sorted histogram
        cat_hist_dict = column.value_counts()
        # Get index of last value in histogram to show
        n_large_cats = largest_category_index_up_to_ratio(cat_hist_dict, max_segments, max_cat_proportions)

        filters = []
        for i in range(n_large_cats):
            f = lambda df, val=cat_hist_dict.index[i]: df[column_name] == val
            filters.append(DeepchecksFilter([f], str(cat_hist_dict.index[i])))

        if len(cat_hist_dict) > n_large_cats:
            f = lambda df, values=cat_hist_dict.index[:n_large_cats]: ~df[column_name].isin(values)
            filters.append(DeepchecksFilter([f], 'Others'))

        return filters


def convert_tree_leaves_into_filters(tree, feature_names: List[str]) -> List[DeepchecksBaseFilter]:
    """Extract the leaves from a sklearn tree and covert them into DeepchecksBaseFilter.

    The function goes over the tree from root to leaf and concatenates (by intersecting) the relevant filters along the
    way. The function returns a list in which each element is a DeepchecksFilter representing the path between the root
    to a different leaf.

    Parameters
    ----------
    tree : A sklearn tree. The tree_ property of a sklearn decision tree.
    feature_names : List[str]
        The feature names for elements within the tree. Normally it is the column names of the data frame the tree
           was trained on.

    Returns
    -------
    List[DeepchecksFilter]:
           A list of filters describing the leaves of the tree.
    """
    node_to_feature = [feature_names[feature_idx] if feature_idx != _tree.TREE_UNDEFINED else None for feature_idx in
                       tree.feature]

    def recurse(node_idx: int, filter_of_node: DeepchecksBaseFilter):
        if tree.feature[node_idx] != _tree.TREE_UNDEFINED:
            left_node_filter = filter_of_node.copy().add_filter(node_to_feature[node_idx], tree.threshold[node_idx],
                                                                greater_then=False)
            right_node_filter = filter_of_node.copy().add_filter(node_to_feature[node_idx], tree.threshold[node_idx])

            return recurse(tree.children_left[node_idx], left_node_filter) + \
                recurse(tree.children_right[node_idx], right_node_filter)
        else:
            return [filter_of_node]

    filters_to_leaves = recurse(0, DeepchecksBaseFilter())
    return filters_to_leaves
