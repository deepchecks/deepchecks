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
import logging
from typing import Union, List

import numpy as np
from PyNomaly import loop

from deepchecks.core import CheckResult, ConditionResult, ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils import gower_distance
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.strings import format_percent, format_number
from deepchecks.utils.typing import Hashable

__all__ = ['OutlierSampleDetection']
logger = logging.getLogger('deepchecks')


class OutlierSampleDetection(SingleDatasetCheck):
    """Detects outliers in a dataset using the LoOP algorithm.

    The LoOP algorithm is a robust method for detecting outliers in a dataset across multiple variables by comparing
    the density in the area of a sample with the densities in the areas of its nearest neighbors.
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
    num_nearest_neighbors : int, default: 10
        Number of nearest neighbors to use for outlier detection.
    extent_parameter: int, default: 3
        Extend parameter for LoOP algorithm.
    n_samples : int , default: 50_000
        number of samples to use for this check.
    n_to_show : int , default: 5
        number of data elements with the highest outlier score to show (out of sample).
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(
            self,
            columns: Union[Hashable, List[Hashable], None] = None,
            ignore_columns: Union[Hashable, List[Hashable], None] = None,
            num_nearest_neighbors: int = 10,
            extent_parameter: int = 3,
            n_samples: int = 50_000,
            n_to_show: int = 5,
            random_state: int = 42,
            **kwargs
    ):
        super().__init__(**kwargs)
        if not isinstance(extent_parameter, int) or extent_parameter <= 0:
            raise DeepchecksValueError('extend_parameter must be a positive integer')
        if not isinstance(num_nearest_neighbors, int) or num_nearest_neighbors <= 0:
            raise DeepchecksValueError('num_nearest_neighbors must be a positive integer')
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.num_nearest_neighbors = num_nearest_neighbors
        self.extent_parameter = extent_parameter
        self.n_samples = n_samples
        self.n_to_show = n_to_show
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_type: str = 'train') -> CheckResult:
        """Run check."""
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test
        dataset = dataset.sample(self.n_samples, random_state=self.random_state, drop_na_label=True)
        df = select_from_dataframe(dataset.data, self.columns, self.ignore_columns)
        if self.num_nearest_neighbors >= len(df):
            logger.warning('Passed num_nearest_neighbors %s which is greater than the number of samples in the dataset'
                           , self.num_nearest_neighbors)
            self.num_nearest_neighbors = len(df) - 1

        # Calculate distances matrix and retrieve nearest neighbors based on distance matrix.
        df_cols_for_gower = df[dataset.cat_features + dataset.numerical_features]
        is_categorical_arr = np.array(df_cols_for_gower.columns.map(lambda x: x in dataset.cat_features), dtype=bool)
        try:
            dist_matrix, idx_matrix = gower_distance.gower_matrix_n_closets(data=np.asarray(df_cols_for_gower),
                                                                            cat_features=is_categorical_arr,
                                                                            num_neighbours=self.num_nearest_neighbors)
        except MemoryError as e:
            raise DeepchecksValueError('A out of memory error occurred while calculating the distance matrix. '
                                       'Try reducing the n_samples or num_nearest_neighbors parameters values.') from e
        # Calculate outlier probability score using loop algorithm.
        m = loop.LocalOutlierProbability(distance_matrix=dist_matrix, neighbor_matrix=idx_matrix,
                                         extent=self.extent_parameter, n_neighbors=self.num_nearest_neighbors).fit()
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

        quantiles_vector = np.quantile(prob_vector, np.array(range(1000)) / 1000, interpolation='nearest')
        return CheckResult(quantiles_vector, display=[headnote, dataset_outliers])

    def add_condition_outlier_ratio_not_greater_than(self, max_outliers_ratio: float = 0.005,
                                                     outlier_score_threshold: float = 0.7):
        """Add condition - no more than given ratio of samples over outlier score threshold are allowed.

        Parameters
        ----------
        max_outliers_ratio : float , default: 0.005
            Maximum ratio of outliers allowed in dataset.
        outlier_score_threshold : float, default: 0.7
            Outlier probability score threshold to be considered outlier.
        """
        if max_outliers_ratio > 1 or max_outliers_ratio < 0:
            raise DeepchecksValueError('max_outliers_ratio must be between 0 and 1')
        name = f'Not more than {format_percent(max_outliers_ratio)} of dataset over ' \
               f'outlier score {format_number(outlier_score_threshold)}'
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

    if quantiles_vector[int(1000 - max_outliers_ratio * 1000)] > outlier_score_threshold:
        ratio_above_threshold = round((1000 - np.argmax(quantiles_vector > outlier_score_threshold)) / 1000, 3)
        details = f'{format_percent(ratio_above_threshold)} of dataset samples above outlier threshold'
        return ConditionResult(ConditionCategory.WARN, details)
    else:
        return ConditionResult(ConditionCategory.PASS)
