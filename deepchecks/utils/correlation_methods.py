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
"""Module containing methods for calculating correlation between features."""

# TODO: get a vector without nulls
from collections import Counter
from typing import Union, List

import math
import numpy as np
import pandas as pd
from scipy.stats import entropy


def conditional_entropy(x: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> float:
    """
        Calculates the conditional entropy of x given y: S(x|y)

        Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
        Parameters:
        -----------
        x: Union[List, np.ndarray, pd.Series]
            A sequence of measurements
        y: Union[List, np.ndarray, pd.Series]
            A sequence of measurements
        Returns:
        --------
        float
            Representing the conditional entropy
    """
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy, math.e)
    return entropy


def theil_u_correlation(x: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> float:
    """
        Calculates the Theil's U correlation of y to x.

        Theil's U is an asymmetric measure ranges [0,1] based on entropy which answers the question: how well does
        feature y explains feature x? For much information see https://en.wikipedia.org/wiki/Uncertainty_coefficient
        Parameters:
        -----------
        x: Union[List, np.ndarray, pd.Series]
            A sequence of measurements
        y: Union[List, np.ndarray, pd.Series]
            A sequence of measurements
        Returns:
        --------
        float
            Representing the conditional entropy
    """
    s_xy = conditional_entropy(x, y)
    x_values_counter = Counter(x)
    total_occurrences = sum(x_values_counter.values())
    values_probabilities = list(map(lambda n: n / total_occurrences, x_values_counter.values()))
    s_x = entropy(values_probabilities)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
