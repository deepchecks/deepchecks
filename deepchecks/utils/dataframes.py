"""Contain functions for handling dataframes in checks."""
from typing import Sequence, Union, List, cast
import pandas as pd
from deepchecks.errors import DeepchecksValueError


__all__ = ['validate_columns_exist', 'filter_columns_with_validation']


def validate_columns_exist(df: pd.DataFrame, columns: Union[None, str, Sequence[str]]):
    """Validate given columns exist in dataframe.

    Args:
        df (pd.DataFrame)
        columns (Union[None, str, List[str]]): Column names to check

    Raise:
        DeepchecksValueError: In case one of columns given don't exists raise error
    """
    if columns is None:
        raise DeepchecksValueError('Got empty columns')
    if isinstance(columns, str):
        columns = [columns]
    elif isinstance(columns, Sequence):
        if any((not isinstance(s, str) for s in columns)):
            raise DeepchecksValueError(f'Columns must be of type str: {", ".join(columns)}')
    else:
        raise DeepchecksValueError(f'Columns must be of types `str` or `List[str]`, but got {type(columns).__name__}')
    # Check columns exists
    non_exists = set(columns) - set(df.columns)
    if non_exists:
        raise DeepchecksValueError(f'Given columns do not exist in dataset: {", ".join(non_exists)}')


def filter_columns_with_validation(df: pd.DataFrame, columns: Union[str, List[str], None] = None,
                                   ignore_columns: Union[str, List[str], None] = None) -> pd.DataFrame:
    """Filter DataFrame columns by given params.

    Args:
        df (pd.DataFrame)
        columns (Union[str, List[str], None]): Column names to keep.
        ignore_columns (Union[str, List[str], None]): Column names to drop.
    Raise:
        DeepchecksValueError: In case one of columns given don't exists raise error
    """
    if columns is not None and ignore_columns is not None:
        raise DeepchecksValueError('Cannot receive both parameters "columns" and "ignore", '
                                   'only one must be used at most')
    elif columns is not None:
        validate_columns_exist(df, columns)
        return cast(pd.DataFrame, df[columns])
    elif ignore_columns is not None:
        validate_columns_exist(df, ignore_columns)
        return df.drop(labels=ignore_columns, axis='columns')
    else:
        return df
