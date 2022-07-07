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
"""module contains the Feature-Feature Correlation check."""

from typing import List, Union

import numpy as np
import pandas as pd
import plotly.express as px

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.correlation_methods import correlation_ratio, symmetric_theil_u_correlation
from deepchecks.utils.dataframes import generalized_corrwith, select_from_dataframe
from deepchecks.utils.typing import Hashable

__all__ = ['FeatureFeatureCorrelation']


class FeatureFeatureCorrelation(SingleDatasetCheck):
    """
    Checks for pairwise correlation between the features.

    Extremely correlated pairs of features could indicate redundancy and even duplication.
    Removing highly correlated features from the data can significantly increase model speed due to the curse of
    dimensionality, and decrease harmful bias.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable.
    show_n_top_columns : int , optional
        amount of columns to show ordered by the highest correlation, default: 10
    n_samples : int , default: 10000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(
        self,
        columns: Union[Hashable, List[Hashable], None] = None,
        ignore_columns: Union[Hashable, List[Hashable], None] = None,
        show_n_top_columns: int = 10,
        n_samples: int = 10000,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.ignore_columns = ignore_columns
        self.n_top_columns = show_n_top_columns
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """
        Run Check.

        Returns
        -------
        CheckResult
            A DataFrame of the pairwise correlations between the features.
        """
        dataset = context.get_data_by_kind(dataset_kind)
        df = select_from_dataframe(dataset.sample(self.n_samples, random_state=self.random_state).data,
                                   self.columns, self.ignore_columns)

        dataset.assert_features()

        # must use list comprehension for deterministic order of columns
        num_features = [f for f in dataset.numerical_features if f in df.columns]
        cat_features = [f for f in dataset.cat_features if f in df.columns]
        encoded_cat_data = df.loc[:, cat_features].apply(lambda x: pd.factorize(x)[0])
        # NaNs are encoded as -1, replace back to NaN
        encoded_cat_data.replace(-1, np.NaN, inplace=True)

        all_features = num_features + cat_features
        full_df = pd.DataFrame(index=all_features, columns=all_features)

        # Numerical-numerical correlations
        if num_features:
            full_df.loc[num_features, num_features] = df.loc[:, num_features].corr(method='spearman')

        # Categorical-categorical correlations
        if cat_features:
            full_df.loc[cat_features, cat_features] = encoded_cat_data.corr(method=symmetric_theil_u_correlation)

        # Numerical-categorical correlations
        if num_features and cat_features:
            num_cat_corr = generalized_corrwith(df.loc[:, num_features], encoded_cat_data,
                                                method=correlation_ratio)
            full_df.loc[num_features, cat_features] = num_cat_corr
            full_df.loc[cat_features, num_features] = num_cat_corr.transpose()

        # Display
        if context.with_display:
            top_n_features = full_df.max(axis=1).sort_values(ascending=False).head(self.n_top_columns).index
            top_n_df = full_df.loc[top_n_features, top_n_features].abs()
            num_nans = top_n_df.isna().sum().sum()
            top_n_df.fillna(0.0, inplace=True)

            fig = [px.imshow(top_n_df, color_continuous_scale=px.colors.sequential.thermal),
                   '* Displayed as absolute values.']
            if num_nans:
                fig.append(f'* NaN values (where the correlation could not be calculated)'
                           f' are displayed as 0.0, total of {num_nans} NaNs in this display.')
            if len(dataset.features) > len(all_features):
                fig.append('* Some features in the dataset are neither numerical nor categorical and therefore not '
                           'calculated.')
        else:
            fig = None

        return CheckResult(value=full_df, header='Feature-Feature Correlation', display=fig)

    def add_condition_max_number_of_pairs_above_threshold(self, threshold: float = 0.9, n_pairs: int = 0):
        """Add condition that all pairwise correlations are less than threshold, except for the diagonal."""
        def condition(result):
            results_ge = result[result > threshold].stack().index.to_list()
            high_corr_pairs = [(i, j) for (i, j) in results_ge if i < j]  # remove diagonal and duplicate pairs

            if len(high_corr_pairs) > n_pairs:
                return ConditionResult(ConditionCategory.FAIL,
                                       f'Correlation is greater than {threshold} for pairs {high_corr_pairs}')
            else:
                return ConditionResult(ConditionCategory.PASS,
                                       f'All correlations are less than {threshold} except pairs {high_corr_pairs}')
        return self.add_condition(f'Not more than {n_pairs} pairs are correlated above {threshold}', condition)
