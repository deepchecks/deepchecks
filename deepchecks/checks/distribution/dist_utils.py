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
"""Common utilities for distribution checks."""

from typing import Tuple, List

from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.impute import SimpleImputer

PSI_MIN_PERCENTAGE = 0.01


__all__ = ['PandasSimpleImputer', 'preprocess_for_psi', 'psi', 'earth_movers_distance', 'drift_score_bar']


def drift_score_bar(axes, drift_score: float, drift_type: str):
    """Create a traffic light bar plot representing the drift score.

    Args:
        axes (): Matplotlib axes object
        drift_score (float): Drift score
        drift_type (str): The name of the drift metric used
    """
    stop = max(0.4, drift_score + 0.1)
    traffic_light_colors = [((0, 0.1), '#01B8AA'),
                            ((0.1, 0.2), '#F2C80F'),
                            ((0.2, 0.3), '#FE9666'),
                            ((0.3, 1), '#FD625E')
                            ]

    for range_tuple, color in traffic_light_colors:
        if range_tuple[0] <= drift_score < range_tuple[1]:
            axes.barh(0, drift_score - range_tuple[0], left=range_tuple[0], color=color)
        elif drift_score >= range_tuple[1]:
            axes.barh(0, range_tuple[1] - range_tuple[0], left=range_tuple[0], color=color)
    axes.set_title('Drift Score - ' + drift_type)
    axes.set_xlim([0, stop])
    axes.set_yticklabels([])


class PandasSimpleImputer(SimpleImputer):
    """A wrapper around `SimpleImputer` to return data frames with columns."""

    def fit(self, X, y=None):  # pylint: disable=C0103
        """Fit the imputer on X and return self."""
        self.columns = X.columns  # pylint: disable=C0103
        return super().fit(X, y)  # pylint: disable=C0103

    def transform(self, X):  # pylint: disable=C0103
        """Transform X using the imputer."""
        return pd.DataFrame(super().transform(X), columns=self.columns)  # pylint: disable=C0103


def preprocess_for_psi(dist1: np.ndarray, dist2: np.ndarray, max_num_categories) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Preprocess distributions in order to be able to be calculated by PSI.

    Function returns the value counts for each distribution and the categories list. If there are more than
    max_num_categories, it encodes rare categories into an "OTHER" category. This is done according to the values of
    dist1, which is treated as the "expected" distribution.

    Function is for categorical data only.
    Args:
        dist1: list of values from the first distribution, treated as the expected distribution
        dist2: list of values from the second distribution, treated as the actual distribution
        max_num_categories: max number of allowed categories. If there are more, they are binned into an "Other"
        category. If max_num_categories=None, there is no limit.

    Returns:
        expected_percents: array of percentages of each value in the expected distribution.
        actual_percents: array of percentages of each value in the actual distribution.
        categories_list: list of all categories that the percentages represent.

    """
    all_categories = list(set(dist1).union(set(dist2)))

    if max_num_categories is not None and len(all_categories) > max_num_categories:
        dist1_counter = dict(Counter(dist1).most_common(max_num_categories))
        dist1_counter['Other rare categories'] = len(dist1) - sum(dist1_counter.values())
        categories_list = list(dist1_counter.keys())

        dist2_counter = Counter(dist2)
        dist2_counter = {k: dist2_counter[k] for k in categories_list}
        dist2_counter['Other rare categories'] = len(dist2) - sum(dist2_counter.values())

    else:
        dist1_counter = Counter(dist1)
        dist2_counter = Counter(dist2)
        categories_list = all_categories

    expected_percents = np.array([dist1_counter[k] for k in categories_list]) / len(dist1)
    actual_percents = np.array([dist2_counter[k] for k in categories_list]) / len(dist2)

    return expected_percents, actual_percents, categories_list


def psi(expected_percents: np.ndarray, actual_percents: np.ndarray):
    """
    Calculate the PSI (Population Stability Index).

    See https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf

    Args:
        expected_percents: array of percentages of each value in the expected distribution.
        actual_percents: array of percentages of each value in the actual distribution.

    Returns:
        psi: The PSI score

    """
    psi_value = 0
    for i in range(len(expected_percents)):
        # In order for the value not to diverge, we cap our min percentage value
        e_perc = max(expected_percents[i], PSI_MIN_PERCENTAGE)
        a_perc = max(actual_percents[i], PSI_MIN_PERCENTAGE)
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        psi_value += value

    return psi_value


def earth_movers_distance(dist1: np.ndarray, dist2: np.ndarray):
    """
    Calculate the Earth Movers Distance (Wasserstein distance).

    See https://en.wikipedia.org/wiki/Wasserstein_metric

    Function is for numerical data only.

    Args:
        dist1: array of numberical values.
        dist2: array of numberical values to compare dist1 to.

    Returns:
        the Wasserstein distance between the two distributions.

    """
    unique1 = np.unique(dist1)
    unique2 = np.unique(dist2)

    sample_space = list(set(unique1).union(set(unique2)))

    val_max = max(sample_space)
    val_min = min(sample_space)

    if val_max == val_min:
        return 0

    dist1 = (dist1 - val_min) / (val_max - val_min)
    dist2 = (dist2 - val_min) / (val_max - val_min)

    return wasserstein_distance(dist1, dist2)
