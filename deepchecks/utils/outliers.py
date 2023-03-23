# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing all outliers algorithms used in the library."""
from typing import Sequence, Tuple, Union

import numpy as np

from deepchecks.core.errors import DeepchecksValueError

EPS = 0.001


def iqr_outliers_range(data: np.ndarray,
                       iqr_range: Tuple[int, int],
                       scale: float) -> Tuple[float, float]:
    """Calculate outliers range on the data given using IQR.

    Parameters
    ----------
    data: np.ndarray
        Data to calculate outliers range for.
    iqr_range: Tuple[int, int]
        Two percentiles which define the IQR range
    scale: float
        The scale to multiply the IQR range for the outliers detection
    min_samples : int, default: 10
        Minimum number of samples needed to calculate outliers
    Returns
    -------
    Tuple[float, float]
        Tuple of lower limit and upper limit of outliers range
    """
    if len(iqr_range) != 2 or any((x < 0 or x > 100 for x in iqr_range)) or all(x < 1 for x in iqr_range):
        raise DeepchecksValueError('IQR range must contain two numbers between 0 to 100')

    q1, q3 = np.percentile(data, sorted(iqr_range))
    iqr = q3 - q1
    low_lim = q1 - scale * iqr
    up_lim = q3 + scale * iqr

    return low_lim, up_lim


def sharp_drop_outliers_range(data_percents: Sequence, sharp_drop_ratio: float = 0.9,
                              max_outlier_percentage: float = 0.05) -> Union[float, None]:
    """Calculate outliers range on the data given using sharp drop.

    Parameters
    ----------
    data_percents : np.ndarray
        Counts of data to calculate outliers range for. The data is assumed to be sorted from the most common to the
        least common.
    sharp_drop_ratio : float , default 0.9
        The sharp drop threshold to use for the outliers detection.
    max_outlier_percentage : float , default 0.05
        The maximum percentage of data that can be considered as "outliers".
    """
    if not 1 - EPS < sum(data_percents) < 1 + EPS:
        raise DeepchecksValueError('Data percents must sum to 1')

    for i in range(len(data_percents) - 1):
        if sum(data_percents[:i+1]) < 1 - max_outlier_percentage:
            continue
        if 1 - (data_percents[i + 1] / data_percents[i]) >= sharp_drop_ratio:
            return data_percents[i + 1]
    else:
        return None
