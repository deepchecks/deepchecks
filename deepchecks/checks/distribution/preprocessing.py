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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from deepchecks.utils.typing import Hashable
from deepchecks.checks.distribution.rare_category_encoder import RareCategoryEncoder


__all__ = ['ScaledNumerics']


class ScaledNumerics:
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

    def __init__(self, baseline_features: pd.DataFrame, categorical_columns: List[Hashable], max_num_categories: int):
        self.cat_columns = categorical_columns
        self.not_cat_columns = list(set(baseline_features.columns) - set(categorical_columns))

        # SimpleImputer doesn't work on all nan-columns, so first replace them  # noqa: SC100
        baseline_features = baseline_features.apply(ScaledNumerics._impute_whole_series_to_zero, axis=0)

        self.categorical_imputer = None
        self.numeric_imputer = None
        if self.cat_columns:
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            self.categorical_imputer.fit(baseline_features[self.cat_columns])
        if self.not_cat_columns:
            self.numeric_imputer = SimpleImputer(strategy='mean')
            self.numeric_imputer.fit(baseline_features[self.not_cat_columns])

        self.scaler = None
        if self.not_cat_columns:
            self.scaler = MinMaxScaler()
            self.scaler.fit(baseline_features[self.not_cat_columns])

        # Replace non-common categories with special value:
        self.rare_category_encoder = RareCategoryEncoder(max_num_categories=max_num_categories,
                                                         cols=categorical_columns)
        self.rare_category_encoder.fit(baseline_features)

        # One-hot encode categorical features:
        self.one_hot_encoder = OneHotEncoder(cols=categorical_columns, use_cat_names=True)
        self.one_hot_encoder.fit(baseline_features)

    def transform(self, features: pd.DataFrame):
        """Transform dataframe based on fit defined when class initialized."""
        # Impute all-nan cols to all-zero:
        features = features.copy()
        features = features.apply(ScaledNumerics._impute_whole_series_to_zero, axis=0)

        if self.categorical_imputer:
            features[self.cat_columns] = self.categorical_imputer.transform(features[self.cat_columns])
        if self.numeric_imputer:
            features[self.not_cat_columns] = self.numeric_imputer.transform(features[self.not_cat_columns])
        if self.scaler:
            features[self.not_cat_columns] = self.scaler.transform(features[self.not_cat_columns])

        features = self.rare_category_encoder.transform(features)
        features = self.one_hot_encoder.transform(features)

        return features

    @staticmethod
    def _impute_whole_series_to_zero(s: pd.Series):
        if s.isna().sum() == s.shape[0]:
            return pd.Series(np.zeros(s.shape))
        else:
            return s
