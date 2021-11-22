"""Utils module containing feature importance calculations."""
import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted

from deepchecks import Dataset
from deepchecks.utils import DeepchecksValueError

__all__ = ['calculate_feature_importance', 'calculate_feature_importance_or_null',
           'column_importance_sorter_dict', 'column_importance_sorter_df']


def calculate_feature_importance_or_null(dataset: Dataset, model: Any) -> pd.Series:
    """Calculate features effect on the label or None if the input is incorrect.

    Args:
        model (Any): A fitted model
        dataset (Dataset): dataset used to fit the model
    Returns:
        pd.Series of feature importance normalized to 0-1 indexed by feature names
        or None if the input is incorrect

    """
    feature_importances = None
    if model and isinstance(dataset, Dataset):
        try:
            # calculate feature importance if dataset has label and the model is fitted on it
            feature_importances = calculate_feature_importance(dataset=dataset, model=model)
        except DeepchecksValueError:
            pass
    return feature_importances


def calculate_feature_importance(model: Any, dataset: Dataset) -> pd.Series:
    """Calculate features effect on the label.

    Args:
        model (Any): A fitted model
        dataset (Dataset): dataset used to fit the model
    Returns:
        pd.Series of feature importance normalized to 0-1 indexed by feature names

    Raise:
        NotFittedError: Call 'fit' with appropriate arguments before using this estimator.
    """
    check_is_fitted(model)
    dataset.validate_model(model)

    if 'feature_importances_' in dir(model):  # Ensambles
        normalized_feature_importance_values = model.feature_importances_/model.feature_importances_.sum()
        feature_importances = pd.Series(normalized_feature_importance_values, index=dataset.features())
    elif 'coef_' in dir(model):  # Linear models
        coef = np.abs(model.coef_)
        coef = coef / coef.sum()
        feature_importances = pd.Series(coef, index=dataset.features())
    else:  # Others
        feature_importances = _calc_importance(model, dataset)

    return feature_importances.fillna(0)


def _calc_importance(model: Any, dataset: Dataset, n_repeats = 30, random_state = 42):
    """Calculate permutation feature importance. Return nonzero value only when std doesn't mask signal."""
    dataset.validate_label('_calc_importance')
    r = permutation_importance(model, dataset.features_columns(),
                               dataset.label_col(),
                               n_repeats=n_repeats,
                               random_state=random_state)
    significance_mask = r.importances_mean - r.importances_std > 0
    feature_importances = r.importances_mean * significance_mask
    total = feature_importances.sum()
    if total != 0:
        feature_importances = feature_importances / total

    return pd.Series(feature_importances, index=dataset.features())


def get_importance(name: str, feature_importances: pd.Series, ds: Dataset):
    """Return importance based on feature importance or label/date/index first."""
    if name in feature_importances.keys():
        return feature_importances[name]
    if name in [ds.label_name(), ds.date_name(), ds.index_name()]:
        return 1
    return 0


def column_importance_sorter_dict(cols_dict: Dict, ds: Dataset, feature_importances: pd.Series,
                                  n_top: int = 10):
    """Return the dict of columns sorted and limited by feature importance.

    Args:
        cols_dict (Dict): dict where columns are the keys
        ds (Dataset): dataset used to fit the model
        feature_importances (pd.Series): feature importance normalized to 0-1 indexed by feature names
        n_top_columns (int): (optinal - used only if model was specified)
                             amount of columns to show ordered by feature importance (date, index, label are first)
    Returns:
        Dict: the dict of columns sorted and limited by feature importance.

    """
    if feature_importances is not None:
        key = lambda name: get_importance(name[0], feature_importances, ds)
        cols_dict = dict(sorted(cols_dict.items(), key=key, reverse=True))
        if n_top:
            return dict(list(cols_dict.items())[:n_top])
    return cols_dict


def column_importance_sorter_df(df: pd.DataFrame, ds: Dataset, feature_importances: pd.Series,
                                n_top: int = 10, col: str = None) -> pd.DataFrame:
    """Return the dataframe of of columns sorted and limited by feature importance.

    Args:
        df (DataFrame): DataFrame to sort
        ds (Dataset): dataset used to fit the model
        feature_importances (pd.Series): feature importance normalized to 0-1 indexed by feature names
        n_top (int): amount of columns to show ordered by feature importance (date, index, label are first)
        col (str): (optional) name of column to sort the dataframe by
    Returns:
        pd.DataFrame: the dataframe sorted and limited by feature importance.

    """
    if feature_importances is not None:
        key = lambda column: [get_importance(name, feature_importances, ds) for name in column]
        if col:
            df = df.sort_values(by=[col], key=key, ascending=False)
        df = df.sort_index(key=key, ascending=False)
        if n_top:
            return df.head(n_top)
    return df
