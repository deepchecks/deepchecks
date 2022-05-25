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
import random
import string

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from deepchecks.utils.array_math import fast_sum_by_row


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


def calculate_nearest_neighbours_distances(cat_data: pd.DataFrame, numeric_data: pd.DataFrame, num_neighbours: int):
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
    cat_data: pd.DataFrame
        The categorical features part of the dataset.
    numeric_data: pd.DataFrame
        The numerical features part of the dataset.
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
    num_samples = cat_data.shape[0]
    num_features = cat_data.shape[1] + numeric_data.shape[1]
    distances, indexes = np.zeros((num_samples, num_neighbours)), np.zeros((num_samples, num_neighbours))
    # handle categorical - transform to an ordinal numpy array
    enc = OrdinalEncoder()
    cat_data = enc.fit_transform(cat_data.fillna(value=''.join(random.choices(string.printable, k=16))))
    # handle numerical - calculate ranges per feature and fill numerical nan to minus np.inf
    numeric_data = np.asarray(numeric_data.fillna(value=np.nan))
    numeric_feature_ranges = np.nanmax(numeric_data, axis=0) - np.nanmin(numeric_data, axis=0)
    numeric_data = np.nan_to_num(numeric_data, nan=np.inf)

    # do not warn on operations that include usage of math involving inf
    original_error_state = np.geterr()['invalid']
    np.seterr(invalid='ignore')

    for i in range(num_samples):  # TODO: parallelize this loop
        dist_to_sample_i = _calculate_distances_to_sample(i, cat_data, numeric_data, numeric_feature_ranges,
                                                          num_features)
        # sort to find the closest samples (including self)
        min_dist_indexes = np.argpartition(dist_to_sample_i, num_neighbours)[:num_neighbours]
        min_dist_indexes_ordered = sorted(min_dist_indexes, key=lambda x, arr=dist_to_sample_i: arr[x], reverse=False)
        indexes[i, :] = min_dist_indexes_ordered
        distances[i, :] = dist_to_sample_i[min_dist_indexes_ordered]

    np.seterr(invalid=original_error_state)
    return np.nan_to_num(distances, nan=np.nan, posinf=np.nan, neginf=np.nan), indexes


def _calculate_distances_to_sample(sample_index: int, cat_data: np.ndarray, numeric_data: np.ndarray,
                                   numeric_feature_ranges: np.ndarray, num_features: int):
    """
    Calculate Gower's distance between a single sample to the rest of the samples in the dataset.

    Parameters
    ----------
    sample_index
        The index of the sample to compare to the rest of the samples.
    cat_data
        The categorical features part of the dataset(after preprocessing).
    numeric_data
        The numerical features part of the dataset(after preprocessing).
    numeric_feature_ranges
        The range sizes of each numerical feature.
    num_features
        The total number of features in the dataset.
    Returns
    -------
    numpy.ndarray
        The distances to the rest of the samples.
    """
    numeric_feat_dist_to_sample = numeric_data - numeric_data[sample_index, :]
    np.abs(numeric_feat_dist_to_sample, out=numeric_feat_dist_to_sample)
    # if a numeric feature value is null for one of the two samples, the distance over it is ignored
    null_dist_locations = np.logical_or(numeric_feat_dist_to_sample == np.inf, numeric_feat_dist_to_sample == np.nan)
    null_numeric_features_per_sample = fast_sum_by_row(null_dist_locations)
    numeric_feat_dist_to_sample[null_dist_locations] = 0
    numeric_feat_dist_to_sample = numeric_feat_dist_to_sample.astype('float64')
    np.divide(numeric_feat_dist_to_sample, numeric_feature_ranges, out=numeric_feat_dist_to_sample)

    cat_feature_dist_to_sample = (cat_data - cat_data[sample_index, :]) != 0

    dist_to_sample = fast_sum_by_row(cat_feature_dist_to_sample) + fast_sum_by_row(numeric_feat_dist_to_sample)
    return dist_to_sample / (-null_numeric_features_per_sample + num_features)  # can have inf values


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
