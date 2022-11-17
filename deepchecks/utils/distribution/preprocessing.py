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
"""Module of preprocessing functions."""
import copy
import warnings
# pylint: disable=invalid-name,unused-argument
from collections import Counter
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from category_encoders import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.distribution.rare_category_encoder import RareCategoryEncoder
from deepchecks.utils.typing import Hashable

__all__ = ['ScaledNumerics', 'preprocess_2_cat_cols_to_same_bins', 'value_frequency']

OTHER_CATEGORY_NAME = 'Other rare categories'


class ScaledNumerics(TransformerMixin, BaseEstimator):
    """Preprocess given features to scaled numerics.

    Parameters
    ----------
    categorical_columns : List[Hashable]
        Indicates names of categorical columns in features.
    max_num_categories : int
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


def preprocess_2_cat_cols_to_same_bins(dist1: Union[np.ndarray, pd.Series], dist2: Union[np.ndarray, pd.Series],
                                       min_category_size_ratio: float = 0., max_num_categories: int = None,
                                       sort_by: str = 'dist1'
                                       ) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Preprocess distributions to the same bins.

    Function returns value counts for each distribution (with the same index) and the categories list.
    Parameter max_num_categories can be used in order to limit the number of resulting bins.

    Function is for categorical data only.

    Parameters
    ----------
    dist1: Union[np.ndarray, pd.Series]
        list of values from the first distribution.
    dist2: Union[np.ndarray, pd.Series]
        list of values from the second distribution.
    min_category_size_ratio: float, default 0
        minimum size ratio for categories. Categories with size ratio lower than this number are binned
        into an "Other" category.
    max_num_categories: int, default: None
        max number of allowed categories. If there are more categories than this number, categories are ordered by
        magnitude and all the smaller categories are binned into an "Other" category.
        If max_num_categories=None, there is no limit.
        > Note that if this parameter is used, the ordering of categories (and by extension, the decision which
        categories are kept by name and which are binned to the "Other" category) is done by default according to the
        values of dist1, which is treated as the "expected" distribution. This behavior can be changed by using the
        sort_by parameter.
    sort_by: str, default: 'dist1'
        Specify how categories should be sorted, affecting which categories will get into the "Other" category.
        Possible values:
        - 'dist1': Sort by the largest dist1 categories.
        - 'dist2': Sort by the largest dist2 categories.
        - 'difference': Sort by the largest difference between categories.
        > Note that this parameter has no effect if max_num_categories = None or there are not enough unique categories.
    Returns
    -------
    dist1_percents
        array of percentages of each value in the first distribution.
    dist2_percents
        array of percentages of each value in the second distribution.
    categories_list
        list of all categories that the percentages represent.

    """
    dist1_counter, dist2_counter = Counter(dist1), Counter(dist2)
    size_dist1, size_dist2 = len(dist1), len(dist2)
    categories_list = list(set(dist1_counter.keys()) | (set(dist2_counter.keys())))

    if max_num_categories is not None and len(categories_list) > max_num_categories:
        if sort_by == 'dist1':
            sort_by_counter = dist1_counter
        elif sort_by == 'dist2':
            sort_by_counter = dist2_counter
        elif sort_by == 'difference':
            sort_by_counter = Counter({key: abs(dist1_counter[key] - dist2_counter[key])
                                       for key in set(dist1_counter.keys()).union(dist2_counter.keys())})
        else:
            raise DeepchecksValueError(f'sort_by got unexpected value: {sort_by}')

        # Not using most_common func of Counter as it's not deterministic for equal values
        categories_list = [x[0] for x in sorted(sort_by_counter.items(), key=lambda x: (-x[1], x[0]))][
                          :max_num_categories]
        dist1_counter = Counter({k: dist1_counter[k] for k in categories_list})
        dist1_counter[OTHER_CATEGORY_NAME] = size_dist1 - sum(dist1_counter.values())
        dist2_counter = Counter({k: dist2_counter[k] for k in categories_list})
        dist2_counter[OTHER_CATEGORY_NAME] = size_dist2 - sum(dist2_counter.values())

    for cat in copy.deepcopy(categories_list):
        if dist1_counter[cat] < size_dist1 * min_category_size_ratio:
            dist1_counter[OTHER_CATEGORY_NAME] += dist1_counter[cat]
            dist2_counter[OTHER_CATEGORY_NAME] += dist2_counter[cat]
            categories_list.remove(cat)

    if dist1_counter[OTHER_CATEGORY_NAME] > (min_category_size_ratio * size_dist1) or \
            dist2_counter[OTHER_CATEGORY_NAME] > (min_category_size_ratio * size_dist2):
        categories_list.append(OTHER_CATEGORY_NAME)

    # aligns both counts to the same index
    dist1_counts = np.array([dist1_counter[k] for k in categories_list])
    dist2_counts = np.array([dist2_counter[k] for k in categories_list])
    return dist1_counts, dist2_counts, categories_list


def value_frequency(x: Union[List, np.ndarray, pd.Series]) -> List[float]:
    """
    Calculate the value frequency of x.

    Parameters:
    -----------
    x: Union[List, np.ndarray, pd.Series]
        A sequence of a categorical variable values.
    Returns:
    --------
    List[float]
        Representing the value frequency of x.
    """
    x_values_counter = Counter(x)
    total_occurrences = len(x)
    values_probabilities = list(map(lambda n: n / total_occurrences, x_values_counter.values()))
    return values_probabilities
