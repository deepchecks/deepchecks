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
"""module contains Correlated Features check."""

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DatasetValidationError
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.correlation_methods import correlation_ratio
from deepchecks.utils.distribution.drift import cramers_v

import pandas as pd

__all__ = ['CorrelatedFeatures']


class CorrelatedFeatures(SingleDatasetCheck):
    """Checks for pairwise correlation between the features.
        Extremely correlated pairs could indicate redundancy and even duplication.

    Parameters
    ----------
        n_top_pairs : int , default: 5
        Number of pairs to show, sorted by the correlation strength
    """
    def __init__(self,
                 n_top_pairs: int = 5,
                 **kwargs):
        super().__init__()
        self.n_top_pairs = n_top_pairs

    def run_logic(self, context: Context, dataset_type: str = 'train') -> CheckResult:
        """
        Run Check

        Returns
        -------
        CheckResult
        A heatmap of the pairwise correlations between the features
        """

        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        dataset.assert_features()
        num_features = dataset.numerical_features
        cat_features = dataset.cat_features

        # Numerical-numerical correlations
        num_num_corr = dataset.data[:, num_features].corr(method='spearman')

        # Numerical-categorical correlations
        num_cat_corr = dataset.data[:, num_features].corrwith(dataset.data[:, cat_features], method=correlation_ratio)

        # Categorical-categorical correlations
        cat_cat_corr = dataset.data[:, cat_features].corr(method=cramers_v)

        # Compose results from all the features
        all_features = num_features + cat_features
        full_df = pd.DataFrame(index=all_features, columns=all_features)
        full_df.loc[num_features, num_features] = num_num_corr
        full_df.loc[cat_features, cat_features] = cat_cat_corr
        full_df.loc[num_features, cat_features] = num_cat_corr
        full_df.loc[cat_features, num_features] = num_cat_corr.transpose()

        return CheckResult(value=full_df)




