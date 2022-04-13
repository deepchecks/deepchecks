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
"""Module for calculating distance matrix via Gower method."""

import numpy as np
import pandas as pd


def gower_matrix(data: np.ndarray, cat_features: np.array) -> np.ndarray:
    """
    Calculate distance matrix for a dataset using Gower's method.

    Gowers distance is a measurement for distance between two samples. It returns the average of their distances
    per feature. For numeric features it calculates the absolute distance divide by the range of the feature. For
    categorical features it is an indicator whether the values are the same.
    See https://www.jstor.org/stable/2528823 for further details. In addition, it can deal with missing values.
    Note that this method is expensive in memory and requires keeping in memory a matrix of size data*data.
    Parameters
    ----------
    data: numpy.ndarray
        Dataset matrix.
    cat_features: numpy.array
        Boolean array of representing which of the columns are categorical features.

    Returns
    -------
    numpy.ndarray
     representing the distance matrix.
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    feature_ranges = np.ones(data.shape[1]) * -1
    feature_ranges[~cat_features] = np.nanmax(data[:, ~cat_features], axis=0) - np.nanmin(data[:, ~cat_features],
                                                                                          axis=0)

    result = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]):
            value = calculate_distance(data[i, :], data[j, :], feature_ranges)
            result[i, j] = value
            result[j, i] = value

    return result


def gower_matrix_n_closets(data: np.ndarray, cat_features: np.array, num_neighbours: int):
    """
    Calculate distance matrix for a dataset using Gower's method.

    Gowers distance is a measurement for distance between two samples. It returns the average of their distances
    per feature. For numeric features it calculates the absolute distance divide by the range of the feature. For
    categorical features it is an indicator whether the values are the same.
    See https://www.jstor.org/stable/2528823 for further details.
    This method minimizes memory usage by saving in memory and returning only the closest neighbours of each sample.
    In addition, it can deal with missing values.
    Parameters
    ----------
    data: numpy.ndarray
        Dataset matrix.
    cat_features: numpy.array
        Boolean array of representing which of the columns are categorical features.
    num_neighbours: int
        Number of neighbours to return. For example, for n=2 for each sample returns the distances to the two closest
        samples in the dataset.
    Returns
    -------
    numpy.ndarray
     representing the distance matrix to the nearest neighbours.
    numpy.ndarray
     representing the indexes of the nearest neighbours.
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    feature_ranges = np.ones(data.shape[1]) * -1
    feature_ranges[~cat_features] = np.nanmax(data[:, ~cat_features], axis=0) - np.nanmin(data[:, ~cat_features],
                                                                                          axis=0)

    distances = np.zeros((data.shape[0], num_neighbours))
    indexes = np.zeros((data.shape[0], num_neighbours))

    for i in range(data.shape[0]):
        dist_to_sample_i = np.zeros(data.shape[0])
        for j in range(data.shape[0]):
            dist_to_sample_i[j] = calculate_distance(data[i, :], data[j, :], feature_ranges)
        # fill na
        dist_to_sample_i[np.isnan(dist_to_sample_i)] = np.nanmean(np.delete(dist_to_sample_i, [i]))
        # sort to find the closest samples
        min_dist_indexes = np.argpartition(dist_to_sample_i, num_neighbours)[:num_neighbours + 1]
        min_dist_indexes_ordered = sorted(min_dist_indexes, key=lambda x, arr=dist_to_sample_i: arr[x], reverse=False)
        indexes[i, :] = min_dist_indexes_ordered[1:]
        distances[i, :] = dist_to_sample_i[min_dist_indexes_ordered[1:]]

    return distances, indexes


def calculate_distance(vec1: np.array, vec2: np.array, range_per_feature: np.array) -> float:
    """Calculate distance between two vectors using Gower's method.

    Parameters
    ----------
    vec1 : np.array
        First vector.
    vec2 : np.array
        Second vector.
    range_per_feature : np.array
        Range of each numeric feature or -1 for categorical.

    Returns
    -------
    float
     representing Gower's distance between the two vectors.
    """
    sum_dist = 0
    num_features = 0
    for col_index in range(len(vec1)):
        if range_per_feature[col_index] == -1:
            # categorical feature
            if pd.isnull(vec1[col_index]) and pd.isnull(vec2[col_index]):
                sum_dist += 0
            elif (pd.isnull(vec1[col_index]) or pd.isnull(vec2[col_index])) or vec1[col_index] != vec2[col_index]:
                sum_dist += 1
            num_features += 1
        else:
            # numeric feature
            if pd.isnull(vec1[col_index]) or pd.isnull(vec2[col_index]):
                continue
            sum_dist += np.abs(vec1[col_index] - vec2[col_index]) / range_per_feature[col_index]
            num_features += 1

    if num_features == 0:
        return np.nan
    return sum_dist / num_features
