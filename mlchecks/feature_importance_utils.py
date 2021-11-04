"""Utils module containing feature importance calculations."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from typing import Any

from mlchecks import Dataset
from mlchecks.utils import MLChecksValueError

__all__ = ['calculate_feature_importance']


def calculate_feature_importance(model: Any, dataset: Dataset) -> [pd.Series]:
    """Calculate features effect on the label.

    Args:
        model (Any): A fitted model
        dataset (Dataset): dataset used to fit the model
    Returns:
        pd.Series of feature importance normalized to 0-1 indexed by feature names

    Raise:
        MLChecksValueError: if model is not fitted or model is not valid for Dataset
    """
    dataset.validate_model(model)

    if 'is_fitted' in dir(model) and model.is_fitted() is False:
        raise MLChecksValueError('Model must be fitted to calculate feature importance')

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
    dataset.validate_label()
    r = permutation_importance(model, dataset.features_columns(),
                               dataset.label_col(),
                               n_repeats=n_repeats,
                               random_state=random_state)
    significance_mask = r.importances_mean - r.importances_std > 0
    feature_importances = r.importances_mean * significance_mask
    return pd.Series(feature_importances, index=dataset.features)
