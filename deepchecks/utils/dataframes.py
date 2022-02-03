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
"""Contain functions for handling dataframes in checks."""
import typing as t

import pandas as pd
import numpy as np

from deepchecks.utils.typing import Hashable
from deepchecks.utils.validation import ensure_hashable_or_mutable_sequence
from deepchecks.errors import DeepchecksValueError


__all__ = ['validate_columns_exist', 'select_from_dataframe', 'un_numpy']


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
    if isinstance(val, np.bool_):
        return str(val)
    if isinstance(val, (np.float64, np.float_)):
        if np.isnan(val):
            return None
        return float(val)
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
        If 'columns' and 'ignore_columns' arguments is 'None'.
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
