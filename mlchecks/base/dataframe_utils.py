"""Contain functions for handling dataframes in checks."""
from typing import List, Union

import pandas as pd

from mlchecks.utils import MLChecksValueError


__all__ = ['validate_columns_exist', 'filter_columns_with_validation']


def validate_columns_exist(df: pd.DataFrame, columns: Union[None, str, List[str]]):
    """Validate given columns exist in dataframe.

    Args:
        df (pd.DataFrame)
        columns (Union[None, str, List[str]]): Column names to check

    Raise:
        MLChecksValueError: In case one of columns given don't exists raise error
    """
    if columns is None:
        raise MLChecksValueError('Got empty columns')
    if isinstance(columns, str):
        columns = [columns]
    elif isinstance(columns, List):
        if any((not isinstance(s, str) for s in columns)):
            raise MLChecksValueError(f'Columns must be of type str: {", ".join(columns)}')
    else:
        raise MLChecksValueError('Columns must be of types `str` or `List[str]`')
    # Check columns exists
    non_exists = set(columns) - set(df.columns)
    if non_exists:
        raise MLChecksValueError(f'Given columns are not exists on dataset: {", ".join(non_exists)}')


def filter_columns_with_validation(df: pd.DataFrame, columns: Union[str, List[str], None] = None,
                                   ignore_columns: Union[str, List[str], None] = None) -> pd.DataFrame:
    """Filter DataFrame columns by given params.

    Args:
        df (pd.DataFrame)
        columns (Union[str, List[str], None]): Column names to keep.
        ignore_columns (Union[str, List[str], None]): Column names to drop.
    Raise:
        MLChecksValueError: In case one of columns given don't exists raise error
    """
    if columns and ignore_columns:
        raise MLChecksValueError('Can\'t have columns and ignore_columns together')
    elif columns:
        validate_columns_exist(df, columns)
        return df[columns]
    elif ignore_columns:
        validate_columns_exist(df, ignore_columns)
        return df.drop(labels=ignore_columns, axis='columns')
    else:
        return df
