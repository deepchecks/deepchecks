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
"""Contain functions for handling dataframes in checks."""
import typing as t

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_integer_dtype, is_numeric_dtype

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.typing import Hashable
from deepchecks.utils.validation import ensure_hashable_or_mutable_sequence

__all__ = ['validate_columns_exist', 'select_from_dataframe', 'un_numpy', 'generalized_corrwith',
           'floatify_dataframe', 'floatify_series', 'default_fill_na_per_column_type']


def default_fill_na_per_column_type(df: pd.DataFrame, cat_features: t.Union[pd.Series, t.List]) -> pd.DataFrame:
    """Fill NaN values per column type."""
    for col_name in df.columns:
        if col_name in cat_features:
            df[col_name].fillna('None', inplace=True)
        elif is_numeric_dtype(df[col_name]):
            df[col_name] = df[col_name].astype('float64').fillna(df[col_name].mean())
        else:
            df[col_name].fillna(df[col_name].mode(), inplace=True)
    return df


def floatify_dataframe(df: pd.DataFrame):
    """Return a dataframe where all the int columns are converted to floats.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to convert

    Raises
    ------
    pd.DataFrame
        the dataframe where all the int columns are converted to floats
    """
    dtype_dict = df.dtypes.to_dict()
    for col_name, dtype in dtype_dict.items():
        if is_integer_dtype(dtype):
            dtype_dict[col_name] = 'float'
    return df.astype(dtype_dict)


def floatify_series(ser: pd.Series):
    """Return a series that if the type is int converted to float.

    Parameters
    ----------
    ser : pd.Series
        series to convert

    Raises
    ------
    pd.Series
        the converted series
    """
    if is_integer_dtype(ser):
        ser = ser.astype(float)
    return ser


def un_numpy(val):
    """Convert numpy value to native value.

    Parameters
    ----------
    val :
        The value to convert.

    Returns
    -------
        returns the numpy value in a native type.
    """
    if isinstance(val, np.generic):
        if np.isnan(val):
            return None
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def validate_columns_exist(
    df: pd.DataFrame,
    columns: t.Union[Hashable, t.List[Hashable]],
    raise_error: bool = True
) -> bool:
    """Validate given columns exist in dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to inspect
    columns : t.Union[Hashable, t.List[Hashable]]
        Column names to check
    raise_error : bool, default: True
        whether to raise an error if some column is not present in the dataframe or not

    Raises
    ------
    DeepchecksValueError
        If some of the columns do not exist within provided dataframe.
        If receives empty list of 'columns'.
        If not all elements within 'columns' list are hashable.
    """
    error_message = 'columns - expected to receive not empty list of hashable values!'
    columns = ensure_hashable_or_mutable_sequence(columns, message=error_message)

    is_empty = len(columns) == 0

    if raise_error and is_empty:
        raise DeepchecksValueError(error_message)
    elif not raise_error and is_empty:
        return False

    difference = set(columns) - set(df.columns)
    all_columns_present = len(difference) == 0

    if raise_error and not all_columns_present:
        stringified_columns = ','.join(map(str, difference))
        raise DeepchecksValueError(f'Given columns do not exist in dataset: {stringified_columns}')

    return all_columns_present


def select_from_dataframe(
    df: pd.DataFrame,
    columns: t.Union[Hashable, t.List[Hashable], None] = None,
    ignore_columns: t.Union[Hashable, t.List[Hashable], None] = None
) -> pd.DataFrame:
    """Filter DataFrame columns by given params.

    Parameters
    ----------
    df : pd.DataFrame
    columns : t.Union[Hashable, t.List[Hashable]] , default: None
        Column names to keep.
    ignore_columns : t.Union[Hashable, t.List[Hashable]] , default: None
        Column names to drop.

    Returns
    -------
    pandas.DataFrame
        returns horizontally filtered dataframe

    Raises
    ------
    DeepchecksValueError
        If some columns do not exist within provided dataframe;
        If 'columns' and 'ignore_columns' arguments are both not 'None'.
    """
    if columns is not None and ignore_columns is not None:
        raise DeepchecksValueError(
            'Cannot receive both parameters "columns" and "ignore", '
            'only one must be used at most'
        )
    elif columns is not None:
        columns = ensure_hashable_or_mutable_sequence(columns)
        validate_columns_exist(df, columns)
        return t.cast(pd.DataFrame, df[columns])
    elif ignore_columns is not None:
        ignore_columns = ensure_hashable_or_mutable_sequence(ignore_columns)
        validate_columns_exist(df, ignore_columns)
        return df.drop(labels=ignore_columns, axis='columns')
    else:
        return df


def generalized_corrwith(x1: pd.DataFrame, x2: pd.DataFrame, method: t.Callable):
    """
    Compute pairwise correlation.

    Pairwise correlation is computed between columns of one DataFrame with columns of another DataFrame.
    Pandas' method corrwith only applies when both dataframes have the same column names,
    this generalized method applies to any two Dataframes with the same number of rows, regardless of the column names.

    Parameters
    ----------
    x1: DataFrame
        Left data frame to compute correlations.
    x2: Dataframe
        Right data frame to compute correlations.
    method: Callable
        Method of correlation. callable with input two 1d ndarrays and returning a float.

    Returns
    -------
    DataFrame
        Pairwise correlations, the index matches the columns of x1 and the columns match the columns of x2.
    """
    corr_results = x2.apply(lambda col: x1.corrwith(col, method=method))
    return corr_results
