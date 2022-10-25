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

"""Utils module containing feature related calculations."""

import typing as t

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_datetime_or_timedelta_dtype, is_float_dtype, is_numeric_dtype

from deepchecks.utils.logger import get_logger
from deepchecks.utils.strings import is_string_column
from deepchecks.utils.typing import Hashable
from deepchecks.utils.validation import ensure_hashable_or_mutable_sequence

__all__ = [
    'infer_categorical_features',
    'infer_numerical_features',
    'is_categorical',
]


def infer_numerical_features(df: pd.DataFrame) -> t.List[Hashable]:
    """Infers which features are numerical.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe for which to infer numerical features

    Returns
    -------
    List[Hashable]
        list of numerical features
    """
    columns = df.columns
    numerical_columns = []
    for col in columns:
        col_data = df[col]
        if col_data.dtype == 'object':
            # object might still be only floats, so we rest the dtype
            col_data = pd.Series(col_data.to_list())
        if is_numeric_dtype(col_data):
            numerical_columns.append(col)
    return numerical_columns


def infer_categorical_features(
        df: pd.DataFrame,
        max_categorical_ratio: float = 0.01,
        max_categories: int = None,
        columns: t.Optional[t.List[Hashable]] = None,
) -> t.List[Hashable]:
    """Infers which features are categorical by checking types and number of unique values.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe for which to infer categorical features
    max_categorical_ratio : float , default: 0.01
    max_categories : int , default: None
    columns : t.Optional[t.List[Hashable]] , default: None

    Returns
    -------
    List[Hashable]
        list of categorical features
    """
    categorical_dtypes = df.select_dtypes(include='category')

    if len(categorical_dtypes.columns) > 0:
        return list(categorical_dtypes.columns)

    if columns is not None:
        dataframe_columns = ensure_hashable_or_mutable_sequence(columns)
    else:
        dataframe_columns = df.columns

    if max_categories is None:
        return [
            column
            for column in dataframe_columns
            if is_categorical(
                t.cast(pd.Series, df[column]),
                max_categorical_ratio)]
    else:
        return [
            column
            for column in dataframe_columns
            if is_categorical(
                t.cast(pd.Series, df[column]),
                max_categorical_ratio,
                max_categories,
                max_categories,
                max_categories)]


def is_categorical(
        column: pd.Series,
        max_categorical_ratio: float = 0.01,
        max_categories_type_string: int = 150,
        max_categories_type_int: int = 30,
        max_categories_type_float_or_datetime: int = 5
) -> bool:
    """Check if uniques are few enough to count as categorical.

    Parameters
    ----------
    column : pd.Series
        A dataframe column
    max_categorical_ratio : float , default: 0.01
    max_categories_type_string : int , default: 150
    max_categories_type_int : int , default: 30
    max_categories_type_float_or_datetime : int , default: 5

    Returns
    -------
    bool
        True if is categorical according to input numbers
    """
    n_samples = len(column.dropna())
    if n_samples == 0:
        get_logger().warning('Column %s only contains NaN values.', column.name)
        return False

    n_samples = np.max([n_samples, 1000])
    n_unique = column.nunique(dropna=True)
    if is_string_column(column):
        return (n_unique / n_samples) < max_categorical_ratio and n_unique <= max_categories_type_string
    elif (is_float_dtype(column) and np.max(column % 1) > 0) or is_datetime_or_timedelta_dtype(column):
        return (n_unique / n_samples) < max_categorical_ratio and n_unique <= max_categories_type_float_or_datetime
    elif is_numeric_dtype(column):
        return (n_unique / n_samples) < max_categorical_ratio and n_unique <= max_categories_type_int
    else:
        return False
