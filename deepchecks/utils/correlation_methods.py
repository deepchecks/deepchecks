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
"""Module containing methods for calculating correlation between variables."""

import math
from collections import Counter
from typing import List, Union

import numpy as np
import pandas as pd
from scipy.stats import entropy

from deepchecks.utils.distribution.preprocessing import value_frequency


def conditional_entropy(x: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> float:
    """
    Calculate the conditional entropy of x given y: S(x|y).

    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    Parameters:
    -----------
    x: Union[List, np.ndarray, pd.Series]
        A sequence of numerical_variable without nulls
    y: Union[List, np.ndarray, pd.Series]
        A sequence of numerical_variable without nulls
    Returns:
    --------
    float
        Representing the conditional entropy
    """
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    s_xy = 0.0
    for xy in xy_counter:
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        s_xy += p_xy * math.log(p_y / p_xy, math.e)
    return s_xy


def theil_u_correlation(x: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> float:
    """
    Calculate the Theil's U correlation of y to x.

    Theil's U is an asymmetric measure ranges [0,1] based on entropy which answers the question: how well does
    variable y explains variable x? For more information see https://en.wikipedia.org/wiki/Uncertainty_coefficient
    Parameters:
    -----------
    x: Union[List, np.ndarray, pd.Series]
        A sequence of a categorical variable values without nulls
    y: Union[List, np.ndarray, pd.Series]
        A sequence of a categorical variable values without nulls
    Returns:
    --------
    float
        Representing the Theil U correlation between y and x
    """
    s_xy = conditional_entropy(x, y)
    values_probabilities = value_frequency(x)
    s_x = entropy(values_probabilities)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def symmetric_theil_u_correlation(x: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> \
        float:
    """
    Calculate the symmetric Theil's U correlation of y to x.

    Parameters:
    -----------
    x: Union[List, np.ndarray, pd.Series]
        A sequence of a categorical variable values without nulls
    y: Union[List, np.ndarray, pd.Series]
        A sequence of a categorical variable values without nulls

    Returns:
    --------
    float
        Representing the symmetric Theil U correlation between y and x
    """
    h_x = entropy(value_frequency(x))
    h_y = entropy(value_frequency(y))
    u_xy = theil_u_correlation(x, y)
    u_yx = theil_u_correlation(y, x)  # pylint: disable=arguments-out-of-order
    u_sym = (h_x * u_xy + h_y * u_yx) / (h_x + h_y)
    return u_sym


def correlation_ratio(categorical_data: Union[List, np.ndarray, pd.Series],
                      numerical_data: Union[List, np.ndarray, pd.Series],
                      ignore_mask: Union[List[bool], np.ndarray] = None) -> float:
    """
    Calculate the correlation ratio of numerical_variable to categorical_variable.

    Correlation ratio is a symmetric grouping based method that describe the level of correlation between
    a numeric variable and a categorical variable. returns a value in [0,1].
    For more information see https://en.wikipedia.org/wiki/Correlation_ratio
    Parameters:
    -----------
    categorical_data: Union[List, np.ndarray, pd.Series]
        A sequence of categorical values encoded as class indices without nulls except possibly at ignored elements
    numerical_data: Union[List, np.ndarray, pd.Series]
        A sequence of numerical values without nulls except possibly at ignored elements
    ignore_mask: Union[List[bool], np.ndarray[bool]] default: None
        A sequence of boolean values indicating which elements to ignore. If None, includes all indexes.
    Returns:
    --------
    float
        Representing the correlation ratio between the variables.
    """
    if ignore_mask:
        numerical_data = numerical_data[~np.asarray(ignore_mask)]
        categorical_data = categorical_data[~np.asarray(ignore_mask)]

    cat_num = int(np.max(categorical_data) + 1)
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(cat_num):
        cat_measures = numerical_data[categorical_data == i]
        n_array[i] = cat_measures.shape[0]
        y_avg_array[i] = np.average(cat_measures.astype(float))  # Cast to float to avoid error in python 3.6
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(numerical_data, y_total_avg), 2))
    if denominator == 0:
        eta = 0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta
