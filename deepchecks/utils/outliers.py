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
                       scale: float,
                       sharp_drop_ratio: float = 0.9) -> Tuple[float, float]:
    """Calculate outliers range on the data given using IQR.

    Parameters
    ----------
    data: np.ndarray
        Data to calculate outliers range for.
    iqr_range: Tuple[int, int]
        Two percentiles which define the IQR range
    scale: float
        The scale to multiply the IQR range for the outliers' detection. When the percentiles values are the same
        (When many samples have the same value),
        the scale will be modified based on the closest element to the percentiles values and
        the `sharp_drop_ratio` parameter.
    sharp_drop_ratio: float, default : 0.9
        A threshold for the sharp drop outliers detection. When more than `sharp_drop_ratio` of the data
        contain the same value the rest will be considered as outliers. Also used to normalize the scale in case
        the percentiles values are the same.
    Returns
    -------
    Tuple[float, float]
        Tuple of lower limit and upper limit of outliers range
    """
    if len(iqr_range) != 2 or any((x < 0 or x > 100 for x in iqr_range)) or all(x < 1 for x in iqr_range):
        raise DeepchecksValueError('IQR range must contain two numbers between 0 to 100')
    if scale < 1:
        raise DeepchecksValueError('IQR scale must be greater than 1')

    q1, q3 = np.percentile(data, sorted(iqr_range))
    if q1 == q3:
        common_percent_in_total = np.sum(data == q1) / len(data)
        if common_percent_in_total > sharp_drop_ratio:
            return q1 - EPS, q1 + EPS
        else:
            closest_dist_to_common = min(np.abs(data[data != q1] - q1))
            # modify the scale to be proportional to the percent of samples that have the same value
            # when many samples have the same value, the scale will be closer to sharp_drop_ratio
            scale = sharp_drop_ratio + ((scale - 1) * (1 - common_percent_in_total))
            return q1 - (closest_dist_to_common * scale), q1 + (closest_dist_to_common * scale)
    else:
        iqr = q3 - q1
        return q1 - scale * iqr, q3 + scale * iqr


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
