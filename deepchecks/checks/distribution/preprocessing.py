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
import numpy as np
import pandas as pd
from typing import List

from category_encoders import OneHotEncoder
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from deepchecks.utils.typing import Hashable
from deepchecks.checks.distribution.rare_category_encoder import RareCategoryEncoder


__all__ = ['ScaledNumerics']


class ScaledNumerics(TransformerMixin):
    """Preprocess given features to scaled numerics.

    Args:
        baseline_features (DataFrame):
            Will be used for fit. Expect to get only features
        categorical_columns (List[Hashable]):
            Indicates names of categorical columns in both DataFrames.
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

    def fit(self, X: pd.DataFrame, **fit_params):
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

        if self.not_cat_columns:
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
        """Transform dataframe based on fit defined when class initialized."""
        # Impute all-nan cols to all-zero:
        X = X.copy()
        X = X.apply(ScaledNumerics._impute_whole_series_to_zero, axis=0)

        if self.categorical_imputer:
            X[self.cat_columns] = self.categorical_imputer.transform(X[self.cat_columns])
        if self.numeric_imputer:
            X[self.not_cat_columns] = self.numeric_imputer.transform(X[self.not_cat_columns])
        if self.scaler:
            X[self.not_cat_columns] = self.scaler.transform(X[self.not_cat_columns])

        X = self.rare_category_encoder.transform(X)
        X = self.one_hot_encoder.transform(X)

        return X

    def fit_transform(self, X, y=None, **fit_params):
        """Fit scaler based on given dataframe and than transform it."""
        self.fit(X, **fit_params)
        return self.transform(X)

    @staticmethod
    def _impute_whole_series_to_zero(s: pd.Series):
        if s.isna().sum() == s.shape[0]:
            return pd.Series(np.zeros(s.shape))
        else:
            return s
