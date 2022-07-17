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
"""Outlier detection functions."""
import time
from typing import List, Union

import numpy as np
from PyNomaly import loop

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import (DeepchecksProcessError, DeepchecksTimeoutError, DeepchecksValueError,
                                    NotEnoughSamplesError)
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils import gower_distance
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.strings import format_number, format_percent
from deepchecks.utils.typing import Hashable

__all__ = ['OutlierSampleDetection']

DATASET_TIME_EVALUATION_SIZE = 100
MINIMUM_NUM_NEAREST_NEIGHBORS = 5


class OutlierSampleDetection(SingleDatasetCheck):
    """Detects outliers in a dataset using the LoOP algorithm.

    The LoOP algorithm is a robust method for detecting outliers in a dataset across multiple variables by comparing
    the density in the area of a sample with the densities in the areas of its nearest neighbors.
    The output of the algorithm is highly dependent on the number of nearest neighbors, it is recommended to
    select a value k that represent the maximum cluster size that will still be considered as "outliers".
    See https://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf for more details.
    LoOP relies on a distance matrix, in our implementation we use the Gower distance that measure the distance
    between two samples based on its numeric and categorical features.
    See https://statisticaloddsandends.wordpress.com/2021/02/23/what-is-gowers-distance/ for further details.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable
    nearest_neighbors_percent : float, default: 0.01
        Percent of the dataset to use as K, nearest neighbors for the LoOP outlier detection. It is recommended to
        select a percentage that represent the maximum cluster size that will still be considered as "outliers".
    extent_parameter: int, default: 3
        Extend parameter for LoOP algorithm.
    n_samples : int , default: 5_000
        number of samples to use for this check.
    n_to_show : int , default: 5
        number of data elements with the highest outlier score to show (out of sample).
    random_state : int, default: 42
        random seed for all check internals.
    timeout : int, default: 10
        Check will be interrupted if it takes more than this number of seconds. If 0, check will not be interrupted.
    """

    def __init__(
            self,
            columns: Union[Hashable, List[Hashable], None] = None,
            ignore_columns: Union[Hashable, List[Hashable], None] = None,
            nearest_neighbors_percent: float = 0.01,
            extent_parameter: int = 3,
            n_samples: int = 5_000,
            n_to_show: int = 5,
            random_state: int = 42,
            timeout: int = 10,
            **kwargs
    ):
        super().__init__(**kwargs)
        if not isinstance(extent_parameter, int) or extent_parameter <= 0:
            raise DeepchecksValueError('extend_parameter must be a positive integer')
        if nearest_neighbors_percent <= 0 or nearest_neighbors_percent > 1:
            raise DeepchecksValueError('nearest_neighbors_percent must be a float between 0 and 1')
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.nearest_neighbors_percent = nearest_neighbors_percent
        self.extent_parameter = extent_parameter
        self.n_samples = n_samples
        self.n_to_show = n_to_show
        self.random_state = random_state
        self.timeout = timeout

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind)
        dataset = dataset.sample(self.n_samples, random_state=self.random_state, drop_na_label=True)
        df = select_from_dataframe(dataset.data, self.columns, self.ignore_columns)
        num_neighbors = int(max(self.nearest_neighbors_percent * df.shape[0], MINIMUM_NUM_NEAREST_NEIGHBORS))
        if df.shape[0] < 1 / self.nearest_neighbors_percent:
            raise NotEnoughSamplesError(
                f'There are not enough samples to run this check, found only {format_number(df.shape[0])} samples.')

        start_time = time.time()
        gower_distance.calculate_nearest_neighbors_distances(
            data=df.iloc[:DATASET_TIME_EVALUATION_SIZE],
            cat_cols=dataset.cat_features,
            numeric_cols=dataset.numerical_features,
            num_neighbors=int(min(np.sqrt(DATASET_TIME_EVALUATION_SIZE), num_neighbors)))
        predicted_time_to_run_in_seconds = ((time.time() - start_time) / 130000) * (df.shape[0] ** 2)
        if predicted_time_to_run_in_seconds > self.timeout > 0:
            raise DeepchecksTimeoutError(
                f'Aborting check: calculation was projected to finish in {predicted_time_to_run_in_seconds} seconds, '
                f'but timeout was configured to {self.timeout} seconds')

        try:
            dist_matrix, idx_matrix = gower_distance.calculate_nearest_neighbors_distances(
                data=df, cat_cols=dataset.cat_features, numeric_cols=dataset.numerical_features,
                num_neighbors=num_neighbors)
        except MemoryError as e:
            raise DeepchecksProcessError('Out of memory error occurred while calculating the distance matrix. Try '
                                         'reducing n_samples or nearest_neighbors_percent parameters values.') from e

        # Calculate outlier probability score using loop algorithm.
        m = loop.LocalOutlierProbability(distance_matrix=dist_matrix, neighbor_matrix=idx_matrix,
                                         extent=self.extent_parameter, n_neighbors=num_neighbors).fit()
        prob_vector = np.asarray(m.local_outlier_probabilities, dtype=float)
        # if we couldn't calculate the outlier probability score for a sample we treat it as not an outlier.
        prob_vector[np.isnan(prob_vector)] = 0

        # Create the check result visualization
        top_n_idx = np.argsort(prob_vector)[-self.n_to_show:]
        dataset_outliers = df.iloc[top_n_idx, :]
        dataset_outliers.insert(0, 'Outlier Probability Score', prob_vector[top_n_idx])
        dataset_outliers.sort_values('Outlier Probability Score', ascending=False, inplace=True)
        headnote = """<span>
                    The Outlier Probability Score is calculated by the LoOP algorithm which measures the local deviation
                    of density of a given sample with respect to its neighbors. These outlier scores are directly
                    interpretable as a probability of an object being an outlier (see
                    <a href="https://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf"
                    target="_blank" rel="noopener noreferrer">link</a> for more information).<br><br>
                    </span>"""

        quantiles_vector = np.quantile(prob_vector, np.array(range(1000)) / 1000, interpolation='higher')
        return CheckResult(quantiles_vector, display=[headnote, dataset_outliers])

    def add_condition_outlier_ratio_less_or_equal(self, max_outliers_ratio: float = 0.005,
                                                  outlier_score_threshold: float = 0.7):
        """Add condition - ratio of samples over outlier score is less or equal to the threshold.

        Parameters
        ----------
        max_outliers_ratio : float , default: 0.005
            Maximum ratio of outliers allowed in dataset.
        outlier_score_threshold : float, default: 0.7
            Outlier probability score threshold to be considered outlier.
        """
        if max_outliers_ratio > 1 or max_outliers_ratio < 0:
            raise DeepchecksValueError('max_outliers_ratio must be between 0 and 1')
        name = f'Ratio of samples exceeding the outlier score threshold {format_number(outlier_score_threshold)} is ' \
               f'less or equal to {format_percent(max_outliers_ratio)}'
        return self.add_condition(name, _condition_outliers_number, outlier_score_threshold=outlier_score_threshold,
                                  max_outliers_ratio=max_outliers_ratio)

    def add_condition_no_outliers(self, outlier_score_threshold: float = 0.7):
        """Add condition - no elements over outlier threshold are allowed.

        Parameters
        ----------
        outlier_score_threshold : float, default: 0.7
            Outlier probability score threshold to be considered outlier.
        """
        name = f'No samples in dataset over outlier score of {format_number(outlier_score_threshold)}'
        return self.add_condition(name, _condition_outliers_number, outlier_score_threshold=outlier_score_threshold)


def _condition_outliers_number(quantiles_vector: np.ndarray, outlier_score_threshold: float,
                               max_outliers_ratio: float = 0):
    max_outliers_ratio = max(round(max_outliers_ratio, 3), 0.001)
    score_at_max_outliers_ratio = quantiles_vector[int(1000 - max_outliers_ratio * 1000)]
    category = ConditionCategory.WARN if score_at_max_outliers_ratio > outlier_score_threshold \
        else ConditionCategory.PASS

    quantiles_above_threshold = quantiles_vector > outlier_score_threshold
    if quantiles_above_threshold.any():
        ratio_above_threshold = round((1000 - np.argmax(quantiles_above_threshold)) / 1000, 3)
    else:
        ratio_above_threshold = 0
    details = f'{format_percent(ratio_above_threshold)} of dataset samples above outlier threshold'

    return ConditionResult(category, details)
