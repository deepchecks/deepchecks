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
# pylint: disable=inconsistent-quotes
"""Utils module containing feature importance calculations."""
import time
import typing as t
from warnings import warn
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_float_dtype
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from deepchecks import base
from deepchecks import errors
from deepchecks.utils import validation
from deepchecks.utils.metrics import get_scorer_single
from deepchecks.utils.typing import Hashable
from deepchecks.utils.model import get_model_of_pipeline


__all__ = [
    'calculate_feature_importance',
    'calculate_feature_importance_or_none',
    'column_importance_sorter_dict',
    'column_importance_sorter_df',
    'infer_categorical_features',
    'is_categorical',
    'N_TOP_MESSAGE'
]


_NUMBER_OF_FEATURES_LIMIT: int = 200
_PERMUTATION_IMPORTANCE_TIMEOUT: int = 120  # seconds
N_TOP_MESSAGE = '* showing only the top %s columns, you can change it using n_top_columns param'


def set_number_of_features_limit(limit: int):
    """Set number of features limit to calculate features importance.

    Args:
        limit (int): limit value
    """
    global _NUMBER_OF_FEATURES_LIMIT
    _NUMBER_OF_FEATURES_LIMIT = limit


def get_number_of_features_limit() -> int:
    """Get number of features limit to calculate features importance."""
    return _NUMBER_OF_FEATURES_LIMIT


def calculate_feature_importance_or_none(
    model: t.Any,
    dataset: t.Union['base.Dataset', pd.DataFrame],
    force_permutation: bool = False,
    permutation_kwargs: t.Optional[t.Dict[str, t.Any]] = None
) -> t.Optional[pd.Series]:
    """Calculate features effect on the label or None if the input is incorrect.

    Args:
        model (Any):
            a fitted model
        dataset (Union[Dataset, pandas.DataFrame]):
            dataset used to fit the model
        force_permutation (bool, default False):
            force permutation importance calculation
        permutation_kwargs (Optional[Dict[str, Any]], defaultNone):
            kwargs for permutation importance calculation

    Returns:
        Optional[pandas.Series]:
            features importance normalized to 0-1 indexed by feature names
            or None if the input is incorrect
    """
    try:
        if model is None:
            return None
        # calculate feature importance if dataset has a label and the model is fitted on it
        return calculate_feature_importance(
            model=model,
            dataset=dataset,
            force_permutation=force_permutation,
            permutation_kwargs=permutation_kwargs
        )
    except (errors.DeepchecksValueError, errors.NumberOfFeaturesLimitError, errors.DeepchecksTimeoutError) as error:
        # DeepchecksValueError:
        #     if model validation failed;
        #     if it was not possible to calculate features importance;
        # NumberOfFeaturesLimitError:
        #     if the number of features limit were exceeded;
        warn(f'Features importance was not calculated:\n{str(error)}')


def calculate_feature_importance(
    model: t.Any,
    dataset: t.Union['base.Dataset', pd.DataFrame],
    force_permutation: bool = False,
    permutation_kwargs: t.Dict[str, t.Any] = None
) -> pd.Series:
    """Calculate features effect on the label.

    Args:
        model (Any):
            a fitted model
        dataset (Union[Dataset, pandas.DataFrame]):
            dataset used to fit the model
        force_permutation (bool, default False):
            force permutation importance calculation
        permutation_kwargs (Optional[Dict[str, Any]], defaultNone):
            kwargs for permutation importance calculation

    Returns:
        pandas.Series: feature importance normalized to 0-1 indexed by feature names

    Raises:
        NotFittedError:
            Call 'fit' with appropriate arguments before using this estimator;
        DeepchecksValueError:
            if model validation failed;
            if it was not possible to calculate features importance;
        NumberOfFeaturesLimitError:
            if the number of features limit were exceeded;
    """
    # TODO: maybe it is better to split it into two functions, one for dataframe instances
    # second for dataset instances
    permutation_kwargs = permutation_kwargs or {}
    permutation_kwargs['random_state'] = permutation_kwargs.get('random_state') or 42
    validation.validate_model(dataset, model)

    if isinstance(dataset, base.Dataset) and force_permutation is True:
        if len(dataset.features) > _NUMBER_OF_FEATURES_LIMIT:
            raise errors.NumberOfFeaturesLimitError(
                f"Dataset contains more than {_NUMBER_OF_FEATURES_LIMIT} of features, "
                "therefore features importance is not calculated. If you want to "
                "change this behaviour please use :function:`deepchecks.utils.features.set_number_of_features_limit`"
            )
        return _calc_importance(model, dataset, **permutation_kwargs).fillna(0)

    feature_importances = _built_in_importance(model, dataset)

    # if _built_in_importance was calculated and returned None,
    # check if pipeline and / or attempt permutation importance
    if feature_importances is None and isinstance(model, Pipeline):
        internal_estimator = get_model_of_pipeline(model)
        if internal_estimator is not None:
            try:
                feature_importances = _built_in_importance(internal_estimator, dataset)
            except ValueError:
                # in case pipeline had an encoder
                pass

    if feature_importances is not None:
        return feature_importances.fillna(0)
    elif isinstance(dataset, base.Dataset):
        return _calc_importance(model, dataset, **permutation_kwargs).fillna(0)
    else:
        raise errors.DeepchecksValueError(
            "Was not able to calculate features importance"  # FIXME: better message
        )


def _built_in_importance(
    model: t.Any,
    dataset: t.Union['base.Dataset', pd.DataFrame],
) -> t.Optional[pd.Series]:
    """Get feature importance member if present in model."""
    features = dataset.features if isinstance(dataset, base.Dataset) else dataset.columns

    if hasattr(model, 'feature_importances_'):  # Ensembles
        normalized_feature_importance_values = model.feature_importances_ / model.feature_importances_.sum()
        return pd.Series(normalized_feature_importance_values, index=features)

    if hasattr(model, 'coef_'):  # Linear models
        coef = np.abs(model.coef_.flatten())
        coef = coef / coef.sum()
        return pd.Series(coef, index=features)


@lru_cache(maxsize=32)
def _calc_importance(
    model: t.Any,
    dataset: 'base.Dataset',
    n_repeats: int = 30,
    mask_high_variance_features: bool = False,
    random_state: int = 42,
    n_samples: int = 10000,
) -> pd.Series:
    """Calculate permutation feature importance. Return nonzero value only when std doesn't mask signal.

    Args:
        model (Any): A fitted model
        dataset (Dataset): dataset used to fit the model
        n_repeats (int): Number of times to permute a feature
        mask_high_variance_features (bool): If true, features for which calculated permutation importance values
                                            varied greatly would be returned has having 0 feature importance
        random_state (int): Random seed for permutation importance calculation.
        n_samples (int): The number of samples to draw from X to compute feature importance
                        in each repeat (without replacement).
    Returns:
        pd.Series of feature importance normalized to 0-1 indexed by feature names
    """
    dataset.validate_label()

    n_samples = min(n_samples, dataset.n_samples)
    dataset_sample_idx = dataset.label_col.sample(n_samples, random_state=random_state).index

    scorer = get_scorer_single(model, dataset, multiclass_avg=False)

    start_time = time.time()
    scorer(model, dataset)
    calc_time = time.time() - start_time

    if calc_time * n_repeats * len(dataset.features) > _PERMUTATION_IMPORTANCE_TIMEOUT:
        raise errors.DeepchecksTimeoutError('Permutation importance calculation was not projected to finish in'
                                            f' {_PERMUTATION_IMPORTANCE_TIMEOUT} seconds.')

    r = permutation_importance(
        model,
        dataset.features_columns.loc[dataset_sample_idx, :],
        dataset.label_col.loc[dataset_sample_idx],
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    significance_mask = (
        r.importances_mean - r.importances_std > 0
        if mask_high_variance_features
        else r.importances_mean > 0
    )

    feature_importances = r.importances_mean * significance_mask
    total = feature_importances.sum()

    if total != 0:
        feature_importances = feature_importances / total

    return pd.Series(feature_importances, index=dataset.features)


def get_importance(name: str, feature_importances: pd.Series, ds: 'base.Dataset') -> int:
    """Return importance based on feature importance or label/date/index first."""
    if name in feature_importances.keys():
        return feature_importances[name]
    if name in [ds.label_name, ds.datetime_name, ds.index_name]:
        return 1
    return 0


def column_importance_sorter_dict(
    cols_dict: t.Dict[Hashable, t.Any],
    dataset: 'base.Dataset',
    feature_importances: t.Optional[pd.Series] = None,
    n_top: int = 10
) -> t.Dict:
    """Return the dict of columns sorted and limited by feature importance.

    Args:
        cols_dict (Dict[Hashable, t.Any]):
            dict where columns are the keys
        dataset (Dataset):
            dataset used to fit the model
        feature_importances (pd.Series):
            feature importance normalized to 0-1 indexed by feature names
        n_top_columns (int):
            amount of columns to show ordered by feature importance (date, index, label are first);
            is used only if model was specified

    Returns:
        Dict[Hashable, Any]: the dict of columns sorted and limited by feature importance.
    """
    if feature_importances is not None:
        key = lambda name: get_importance(name[0], feature_importances, dataset)
        cols_dict = dict(sorted(cols_dict.items(), key=key, reverse=True))
        if n_top:
            return dict(list(cols_dict.items())[:n_top])
    return cols_dict


def column_importance_sorter_df(
    df: pd.DataFrame,
    ds: 'base.Dataset',
    feature_importances: pd.Series,
    n_top: int = 10,
    col: t.Optional[Hashable] = None
) -> pd.DataFrame:
    """Return the dataframe of of columns sorted and limited by feature importance.

    Args:
        df (DataFrame): DataFrame to sort
        ds (Dataset): dataset used to fit the model
        feature_importances (pd.Series): feature importance normalized to 0-1 indexed by feature names
        n_top (int): amount of columns to show ordered by feature importance (date, index, label are first)
        col (Optional[Hashable]): name of column to sort the dataframe by
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


def infer_categorical_features(
    df: pd.DataFrame,
    max_categorical_ratio: float = 0.01,
    max_categories: int = 30,
    max_float_categories: int = 5,
    columns: t.Optional[t.List[Hashable]] = None,
) -> t.List[Hashable]:
    """Infers which features are categorical by checking types and number of unique values.

    Arguments:
        df (DataFrame): dataframe for which to infer categorical features

    Returns:
        List[hashable]: list of categorical features
    """
    categorical_dtypes = df.select_dtypes(include='category')

    if len(categorical_dtypes.columns) > 0:
        return list(categorical_dtypes.columns)

    if columns is not None:
        dataframe_columns = validation.ensure_hashable_or_mutable_sequence(columns)
    else:
        dataframe_columns = df.columns

    return [
        column
        for column in dataframe_columns
        if is_categorical(
            t.cast(pd.Series, df[column]),
            max_categorical_ratio,
            max_categories,
            max_float_categories
        )
    ]


def is_categorical(
    column: pd.Series,
    max_categorical_ratio: float = 0.01,
    max_categories: int = 30,
    max_float_categories: int = 5
) -> bool:
    """Check if uniques are few enough to count as categorical.

    Args:
        column (Series):
            The name of the column in the dataframe

    Returns:
        bool: True if is categorical according to input numbers
    """
    n_unique = column.nunique(dropna=True)
    n_samples = len(column.dropna())

    if is_float_dtype(column):
        return n_unique <= max_float_categories

    return n_unique / n_samples < max_categorical_ratio and n_unique <= max_categories
