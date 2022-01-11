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
"""Module of preprocessing functions."""
# pylint: disable=invalid-name,unused-argument
from collections import Counter

import numpy as np
import pandas as pd
from typing import List, Tuple

from category_encoders import OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from deepchecks.utils.typing import Hashable
from deepchecks.utils.distribution.rare_category_encoder import RareCategoryEncoder


__all__ = ['ScaledNumerics', 'preprocess_2_cat_cols_to_same_bins']


class ScaledNumerics(TransformerMixin, BaseEstimator):
    """Preprocess given features to scaled numerics.

    Args:
        categorical_columns (List[Hashable]):
            Indicates names of categorical columns in features.
        max_num_categories (int):
            Indicates the maximum number of unique categories in a single categorical column
            (rare categories will be changed to a form of "other")
    """

    def __init__(self, categorical_columns: List[Hashable], max_num_categories: int):
        self.one_hot_encoder = None
        self.rare_category_encoder = None
        self.scaler = None
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.not_cat_columns = None
        self.cat_columns = categorical_columns
        self.max_num_categories = max_num_categories

    def fit(self, X: pd.DataFrame):
        """Fit scaler based on given dataframe."""
        self.not_cat_columns = list(set(X.columns) - set(self.cat_columns))

        # SimpleImputer doesn't work on all nan-columns, so first replace them  # noqa: SC100
        X = X.apply(ScaledNumerics._impute_whole_series_to_zero, axis=0)

        if self.cat_columns:
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            self.categorical_imputer.fit(X[self.cat_columns])
        if self.not_cat_columns:
            self.numeric_imputer = SimpleImputer(strategy='mean')
            self.numeric_imputer.fit(X[self.not_cat_columns])
            self.scaler = MinMaxScaler()
            self.scaler.fit(X[self.not_cat_columns])

        # Replace non-common categories with special value:
        self.rare_category_encoder = RareCategoryEncoder(max_num_categories=self.max_num_categories,
                                                         cols=self.cat_columns)
        self.rare_category_encoder.fit(X)

        # One-hot encode categorical features:
        self.one_hot_encoder = OneHotEncoder(cols=self.cat_columns, use_cat_names=True)
        self.one_hot_encoder.fit(X)

    def transform(self, X: pd.DataFrame):
        """Transform features into scaled numerics."""
        # Impute all-nan cols to all-zero:
        X = X.copy()
        X = X.apply(ScaledNumerics._impute_whole_series_to_zero, axis=0)

        if self.cat_columns:
            X[self.cat_columns] = self.categorical_imputer.transform(X[self.cat_columns])
        if self.not_cat_columns:
            X[self.not_cat_columns] = self.numeric_imputer.transform(X[self.not_cat_columns])
            X[self.not_cat_columns] = self.scaler.transform(X[self.not_cat_columns])

        X = self.rare_category_encoder.transform(X)
        X = self.one_hot_encoder.transform(X)

        return X

    def fit_transform(self, X, y=None, **fit_params):
        """Fit scaler based on given dataframe and then transform it."""
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def _impute_whole_series_to_zero(s: pd.Series):
        """If given series contains only nones, return instead series with only zeros."""
        if s.isna().all():
            return pd.Series(np.zeros(s.shape))
        else:
            return s


def preprocess_2_cat_cols_to_same_bins(dist1: np.ndarray, dist2: np.ndarray, max_num_categories
                                       ) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Preprocess distributions to the same bins in order to be able to be calculated by PSI.

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
        dist1_percents: array of percentages of each value in the first distribution.
        dist2_percents: array of percentages of each value in the second distribution.
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

    dist1_percents = np.array([dist1_counter[k] for k in categories_list]) / len(dist1)
    dist2_percents = np.array([dist2_counter[k] for k in categories_list]) / len(dist2)

    return dist1_percents, dist2_percents, categories_list
