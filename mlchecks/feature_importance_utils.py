"""Utils module containing feature importance calculations."""
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted

from mlchecks import Dataset
from mlchecks.utils import MLChecksValueError

__all__ = ['calculate_feature_importance', 'calculate_feature_importance_or_null', 'column_importance_sorter']


def calculate_feature_importance_or_null(dataset: Dataset, model: Any) -> pd.Series:
    feature_importances = None
    if model:
        try:
            feature_importances = calculate_feature_importance(dataset=dataset, model=model)
        except MLChecksValueError:
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


def _calc_importance(model: Any, dataset: Dataset, n_repeats=30, random_state=42):
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
    if name in feature_importances:
        return feature_importances[name]
    if name in [ds.label_name(), ds.date_name(), ds.index_name()]:
        return 1
    return 0


def column_importance_sorter_dict(cols_dict: Dict, ds: Dataset, feature_importances: pd.Series,
                             n_top: int=10): 
    if feature_importances:
        key = lambda col: get_importance(col, feature_importances, ds)
        cols_dict = dict(sorted(cols_dict.items(), key=key))
    if n_top:
        return dict(list(cols_dict.items())[:n_top])
    return cols_dict


def column_importance_sorter_df(df: pd.DataFrame, ds: Dataset, feature_importances: pd.Series,
                             n_top: int=10, cols: List[str]=None): 
    if feature_importances:
        key = lambda col: get_importance(col, feature_importances, ds)
        if cols:
            df = df.sort_values(by=cols, key=key)
        df = df.sort_index(key=key)
    if n_top:
        return df.head(n_top)
    return df
