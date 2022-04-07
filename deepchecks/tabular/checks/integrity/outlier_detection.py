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
from typing import Union, List

import numpy as np
from PyNomaly import loop

from deepchecks.core import CheckResult, ConditionResult, ConditionCategory
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.dataframes import select_from_dataframe
from deepchecks.utils.typing import Hashable

__all__ = ['OutlierDetection']


class OutlierDetection(SingleDatasetCheck):
    """Detects outliers in a dataset using the LoOP algorithm.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable
    num_nearest_neighbors : int, default: 20
        Number of nearest neighbors to use for outlier detection.
    extend_parameter: int, default: 3
        Extend parameter for LoOP algorithm.
    n_to_show : int , default: 5
        number of most data elements with the highest outlier score to show.
    """

    def __init__(
            self,
            columns: Union[Hashable, List[Hashable], None] = None,
            ignore_columns: Union[Hashable, List[Hashable], None] = None,
            num_nearest_neighbors: int = 20,
            extend_parameter: int = 3,
            n_to_show: int = 5,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.num_nearest_neighbors = num_nearest_neighbors
        self.extend_parameter = extend_parameter
        self.n_to_show = n_to_show

    def run_logic(self, context: Context, dataset_type: str = 'train') -> CheckResult:
        """Run check."""
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        df = select_from_dataframe(dataset.data, self.columns, self.ignore_columns)

        m = loop.LocalOutlierProbability(df, extent=self.extend_parameter, n_neighbors=self.num_nearest_neighbors).fit()
        prob_vector = m.local_outlier_probabilities

        # Create the check result visualization
        top_n_idx = np.argsort(prob_vector)[-self.n_to_show:]
        top_n_values = [prob_vector[i] for i in top_n_idx]
        dataset_outliers = dataset.iloc[top_n_idx, :]
        dataset_outliers.insert(0, 'Outlier Probability Score', top_n_values)
        dataset_outliers.sort_values('Outlier Probability Score', ascending=False, inplace=True)

        headnote = """<span>
                    The Outlier Probability Score is calculated by the LoOP algorithm which measures the local deviation
                     of density of a given sample with respect to its neighbors. These outlier scores are directly
                      interpretable as a probability of an object being an outlier. see 
                      <a href="https://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf" target="_blank" rel="noopener noreferrer">link</a>
                       for more information.<br><br>
                </span>"""

        quantiles_vector = np.quantile(prob_vector, np.array(range(1000)) / 1000)
        return CheckResult([quantiles_vector, df.shape[0]], display=[headnote, dataset_outliers])

    def add_condition_not_more_outliers_than(self, num_max_outliers: int = 0, max_outliers_ratio: float = 0.005,
                                             outlier_score_threshold: float = 0.8):
        """Add condition - no more than given number of elements over outlier threshold are allowed.

        Parameters
        ----------
        num_max_outliers : int , default: 0
            Maximum number of outliers allowed.
        max_outliers_ratio : float , default: 0.005
            Maximum ratio of outliers allowed.
        outlier_score_threshold : float, default: 0.8
            Outlier score threshold to use.
        """
        if max_outliers_ratio > 1 or max_outliers_ratio < 0:
            raise ValueError('max_outliers_ratio must be between 0 and 1')
        if num_max_outliers < 0:
            raise ValueError('num_max_outliers must be a positive integer')
        if num_max_outliers > 0:
            if max_outliers_ratio != 0.005:
                raise ValueError('Only one of max_outliers_ratio and num_max_outliers can be given')
            else:
                name = f'Not more than {num_max_outliers} outliers in dataset over score threshold {outlier_score_threshold}'
        else:
            name = f'Not more than {max_outliers_ratio} of outlier ratio in dataset over threshold {outlier_score_threshold}'
        return self.add_condition(name, _condition_outliers_number, outlier_score_threshold=outlier_score_threshold,
                                  num_max_outliers=num_max_outliers, max_outliers_ratio=max_outliers_ratio)

    def add_condition_no_outliers(self, outlier_score_threshold: float = 0.8):
        """Add condition - no elements over outlier threshold are allowed.

        Parameters
        ----------
        outlier_score_threshold : float, default: 0.8
            Outlier score threshold to use.
        """
        name = f'No outliers in dataset over score threshold {outlier_score_threshold}'
        return self.add_condition(name, _condition_outliers_number, outlier_score_threshold=outlier_score_threshold)


def _condition_outliers_number(result: np.ndarray, outlier_score_threshold: float, num_max_outliers: int = 0,
                               max_outliers_ratio: float = 0):
    if num_max_outliers > 0:
        max_outliers_ratio = num_max_outliers / result[1]
    max_outliers_ratio = round(max_outliers_ratio, 3)
    quantiles_vector = result[0]

    if quantiles_vector[int(1000 - max_outliers_ratio * 1000)] > outlier_score_threshold:
        approx_num_above_threshold = round(
            ((1000 - np.argmax(quantiles_vector > outlier_score_threshold)) / 1000) * result[1])
        details = f'Found {approx_num_above_threshold} elements above outlier threshold of {outlier_score_threshold}'
        return ConditionResult(ConditionCategory.WARN, details)
    else:
        return ConditionResult(ConditionCategory.PASS)
