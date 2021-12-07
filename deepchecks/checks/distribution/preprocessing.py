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
from typing import List, Tuple

from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from deepchecks.utils.typing import Hashable
from deepchecks.checks.distribution.rare_category_encoder import RareCategoryEncoder


__all__ = ['preprocess_dataset_to_scaled_numerics']


def preprocess_dataset_to_scaled_numerics(
    baseline_features: pd.DataFrame,
    test_features: pd.DataFrame,
    categorical_columns: List[Hashable],
    max_num_categories: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess given features to scaled numerics.

    Args:
        baseline_features (DataFrame):
            Will be used for fit and also transformed. Expect to get only features
        test_features (DataFrame):
            Will be transformed according to baseline_data. Expect to get only features
        categorical_columns (List[Hashable]):
            Indicates names of categorical columns in both DataFrames.
        max_num_categories (int):
            Indicates the maximum number of unique categories in a single categorical column
            (rare categories will be changed to a form of "other")

    Returns:
        (DataFrame, DataFrame): returns both datasets transformed.
    """
    x_baseline = baseline_features.copy()
    x_test = test_features.copy()
    non_categorical_columns = list(set(test_features.columns) - set(categorical_columns))

    # Impute all-nan cols to all-zero:
    def impute_whole_series_to_zero(s: pd.Series):
        if s.isna().sum() == s.shape[0]:
            return pd.Series(np.zeros(s.shape))
        else:
            return s

    x_baseline = x_baseline.apply(impute_whole_series_to_zero, axis=0)
    x_test = x_test.apply(impute_whole_series_to_zero, axis=0)

    # impute nan values:
    if x_baseline.isna().any().any():
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        numeric_imputer = SimpleImputer(strategy='mean')
        if categorical_columns:
            x_baseline[categorical_columns] = categorical_imputer.fit_transform(x_baseline[categorical_columns])
            x_test[categorical_columns] = categorical_imputer.transform(x_test[categorical_columns])
        if non_categorical_columns:
            x_baseline[non_categorical_columns] = numeric_imputer.fit_transform(x_baseline[non_categorical_columns])
            x_test[non_categorical_columns] = numeric_imputer.transform(x_test[non_categorical_columns])

    # Scale numeric features between 0 and 1:
    scaler = MinMaxScaler()
    if non_categorical_columns:
        x_baseline[non_categorical_columns] = scaler.fit_transform(x_baseline[non_categorical_columns])
        x_test[non_categorical_columns] = scaler.transform(x_test[non_categorical_columns])

    # Replace non-common categories with special value:
    rare_category_encoder = RareCategoryEncoder(max_num_categories=max_num_categories, cols=categorical_columns)
    x_baseline = rare_category_encoder.fit_transform(x_baseline)
    x_test = rare_category_encoder.transform(x_test)

    # One-hot encode categorical features:
    one_hot_encoder = OneHotEncoder(cols=categorical_columns, use_cat_names=True)
    x_baseline = one_hot_encoder.fit_transform(x_baseline)
    x_test = one_hot_encoder.transform(x_test)

    return x_baseline, x_test
