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
"""module contains  Feature-Feature Correlation check."""

import pandas as pd
import plotly.express as px

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.correlation_methods import correlation_ratio, symmetric_theil_u_correlation
from deepchecks.utils.dataframes import generalized_corrwith

__all__ = ['FeatureFeatureCorrelation']


class FeatureFeatureCorrelation(SingleDatasetCheck):
    """
    Checks for pairwise correlation between the features.

    Extremely correlated pairs could indicate redundancy and even duplication.
    """

    def __init__(self,
                 **kwargs):
        super().__init__()

    def run_logic(self, context: Context, dataset_type: str = 'train') -> CheckResult:
        """
        Run Check.

        Returns
        -------
        CheckResult
            A DataFrame of the pairwise correlations between the features.
        """
        if dataset_type == 'train':
            dataset = context.train
        else:
            dataset = context.test

        dataset.assert_features()
        num_features = dataset.numerical_features
        cat_features = dataset.cat_features

        encoded_cat_data = dataset.data.loc[:, cat_features].apply(lambda x: pd.factorize(x)[0])

        # Numerical-numerical correlations
        num_num_corr = dataset.data.loc[:, num_features].corr(method='spearman')

        # Numerical-categorical correlations
        num_cat_corr = generalized_corrwith(dataset.data.loc[:, num_features], encoded_cat_data,
                                            method=correlation_ratio)

        # Categorical-categorical correlations
        cat_cat_corr = encoded_cat_data.corr(method=symmetric_theil_u_correlation)

        # Compose results from all the features
        all_features = num_features + cat_features
        full_df = pd.DataFrame(index=all_features, columns=all_features)
        full_df.loc[num_features, num_features] = num_num_corr
        full_df.loc[cat_features, cat_features] = cat_cat_corr
        full_df.loc[num_features, cat_features] = num_cat_corr
        full_df.loc[cat_features, num_features] = num_cat_corr.transpose()

        # Display
        fig = px.imshow(full_df)

        return CheckResult(value=full_df, header='Feature-Feature Correlation', display=fig)

    def add_condition_max_number_of_pairs_above(self, threshold: float = 0.9, n_pairs: int = 0):
        """Add condition that all pairwise correlations are less than threshold, except for the diagonal."""
        def condition(result):
            results_ge = result.ge(threshold)
            high_corr_pairs = []
            for i in results_ge.index:
                for j in results_ge.columns:
                    if (i == j) | ((j, i) in high_corr_pairs):
                        continue
                    if results_ge.loc[i, j]:
                        high_corr_pairs.append((i, j))

            if len(high_corr_pairs) > n_pairs:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Correlation is greater than {threshold} for pairs {high_corr_pairs}')
            else:
                return ConditionResult(ConditionCategory.PASS,
                                       f'All correlations are less than {threshold} except pairs {high_corr_pairs}')
        return self.add_condition(f'Not more than {n_pairs} pairs are correlated above {threshold}', condition)
