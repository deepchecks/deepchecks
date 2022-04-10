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

    Can deal with missing values.
    Requires each two samples to have at least one sheared non-null value.
    Parameters
    ----------
    data: numpy.ndarray
        Dataset matrix.
    cat_features: numpy.array
        Boolean array of representing which of the columns are categorical features.

    Returns numpy.ndarray representing the distance matrix.
    """
    if not isinstance(data, np.ndarray): data = np.asarray(data)
    ranges = calculate_ranges(data, cat_features)

    result = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]):
            value = calculate_distance(data[i, :], data[j, :], ranges)
            result[i, j] = value
            result[j, i] = value

    return result


def calculate_ranges(data: np.ndarray, cat_features: np.array) -> np.array:
    """Calculate ranges for each numeric feature and returns -1 for other features.

    Parameters
    ----------
    data: numpy.ndarray
        Dataset matrix.
    cat_features: numpy.array
        Boolean array of representing which of the columns are categorical features.

    Returns numpy.array representing the ranges of each numeric feature and -1 for categorical.
    """
    ranges = np.zeros(data.shape[1])
    for col_index in range(data.shape[1]):
        if cat_features[col_index]:
            ranges[col_index] = -1
        else:
            ranges[col_index] = max(data[:, col_index]) - min(data[:, col_index])
    return ranges


def calculate_distance(vec1: np.array, vec2: np.array, ranges: np.array) -> float:
    """Calculate distance between two vectors using Gower's method.

    Parameters
    ----------
        vec1: First vector.
        vec2: Second vector.
        ranges: Ranges of each numeric feature or -1 for categorical.

    Returns float representing Gower's distance between two vectors.
    """
    sum_dist = 0
    num_features = 0
    for col_index in range(len(vec1)):
        if ranges[col_index] == -1:
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
            sum_dist += np.abs(vec1[col_index] - vec2[col_index]) / ranges[col_index]
            num_features += 1

    if num_features == 0:
        raise ValueError(f"No non null features found to compare examples {vec1} and {vec2}.")
    return sum_dist / num_features
