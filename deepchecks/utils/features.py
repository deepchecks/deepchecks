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
import warnings
from warnings import warn
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_float_dtype
from sklearn.inspection import permutation_importance

from deepchecks import base
from deepchecks import errors
from deepchecks.utils import validation
from deepchecks.utils.metrics import DeepcheckScorer, get_default_scorers, task_type_check, init_validate_scorers
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


N_TOP_MESSAGE = '* showing only the top %s columns, you can change it using n_top_columns param'


def calculate_feature_importance_or_none(
    model: t.Any,
    dataset: t.Union['base.Dataset', pd.DataFrame],
    force_permutation: bool = False,
    permutation_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Tuple[t.Optional[pd.Series], t.Optional[str]]:
    """Calculate features effect on the label or None if the input is incorrect.

    Parameters
    ----------
    model : t.Any
        a fitted model
    dataset : t.Union['base.Dataset', pd.DataFrame]
        dataset used to fit the model
    force_permutation : bool , default: False
        force permutation importance calculation
    permutation_kwargs : t.Optional[t.Dict[str, t.Any]] , default: None
        kwargs for permutation importance calculation

    Returns
    -------
    feature_importance, calculation_type : t.Tuple[t.Optional[pd.Series], str]]
        features importance normalized to 0-1 indexed by feature names, or None if the input is incorrect
        Tuple of the features importance and the calculation type
        (types: `permutation_importance`, `feature_importances_`, `coef_`)
    """
    try:
        if model is None:
            return None
        # calculate feature importance if dataset has a label and the model is fitted on it
        fi, calculation_type = calculate_feature_importance(
            model=model,
            dataset=dataset,
            force_permutation=force_permutation,
            permutation_kwargs=permutation_kwargs,
        )

        return fi, calculation_type
    except (
        errors.DeepchecksValueError,
        errors.NumberOfFeaturesLimitError,
        errors.DeepchecksTimeoutError,
        errors.ModelValidationError,
        errors.DatasetValidationError
    ) as error:
        # DeepchecksValueError:
        #     if model validation failed;
        #     if it was not possible to calculate features importance;
        # NumberOfFeaturesLimitError:
        #     if the number of features limit were exceeded;
        # DatasetValidationError:
        #     if dataset did not meet requirements
        # ModelValidationError:
        #     if wrong type of model was provided;
        #     if function failed to predict on model;
        warn(f'Features importance was not calculated:\n{str(error)}')
        return None, None


def calculate_feature_importance(
    model: t.Any,
    dataset: t.Union['base.Dataset', pd.DataFrame],
    force_permutation: bool = False,
    permutation_kwargs: t.Dict[str, t.Any] = None,
) -> t.Tuple[pd.Series, str]:
    """Calculate features effect on the label.

    Parameters
    ----------
    model : t.Any
        a fitted model
    dataset : t.Union['base.Dataset', pd.DataFrame]
        dataset used to fit the model
    force_permutation : bool, default: False
        force permutation importance calculation
    permutation_kwargs : t.Dict[str, t.Any] , default: None
        kwargs for permutation importance calculation

    Returns
    -------
    Tuple[Series, str]:
        first item - feature importance normalized to 0-1 indexed by feature names,
        second item - type of feature importance calculation (types: `permutation_importance`,
        `feature_importances_`, `coef_`)

    Raises
    ------
    NotFittedError
        Call 'fit' with appropriate arguments before using this estimator.
    DeepchecksValueError
        if model validation failed.
        if it was not possible to calculate features importance.
    NumberOfFeaturesLimitError
        if the number of features limit were exceeded.
    """
    # TODO: maybe it is better to split it into two functions, one for dataframe instances
    # second for dataset instances
    permutation_kwargs = permutation_kwargs or {}
    permutation_kwargs['random_state'] = permutation_kwargs.get('random_state') or 42
    validation.validate_model(dataset, model)
    permutation_failure = None
    calc_type = None
    importance = None

    if force_permutation:
        if isinstance(dataset, pd.DataFrame):
            permutation_failure = 'Cannot calculate permutation feature importance on dataframe, using' \
                                  ' built-in model\'s feature importance instead'
        else:
            try:
                importance = _calc_permutation_importance(model, dataset, **permutation_kwargs)
                calc_type = 'permutation_importance'
            except errors.DeepchecksTimeoutError as e:
                permutation_failure = f'{e.message}\n using model\'s built-in feature importance instead'

    # If there was no force permutation, or it failed tries to take importance from the model
    if importance is None:
        # Get the actual model in case of pipeline
        internal_estimator = get_model_of_pipeline(model)
        importance, calc_type = _built_in_importance(internal_estimator, dataset)
        # If found importance and was force permutation failure before, show warning
        if importance is not None and permutation_failure:
            warnings.warn(permutation_failure)

    # If there was no permutation failure and no importance on the model, using permutation anyway
    if importance is None and permutation_failure is None and isinstance(dataset, base.Dataset):
        importance = _calc_permutation_importance(model, dataset, **permutation_kwargs)
        calc_type = 'permutation_importance'
        warnings.warn('Could not find built-in feature importance on the model, using '
                      'permutation feature importance calculation')

    # If after all importance is still none raise error
    if importance is None:
        # FIXME: better message
        raise errors.DeepchecksValueError("Was not able to calculate features importance")
    return importance.fillna(0), calc_type


def _built_in_importance(
    model: t.Any,
    dataset: t.Union['base.Dataset', pd.DataFrame],
) -> t.Tuple[t.Optional[pd.Series], t.Optional[str]]:
    """Get feature importance member if present in model."""
    features = dataset.features if isinstance(dataset, base.Dataset) else dataset.columns

    try:
        if hasattr(model, 'feature_importances_'):  # Ensembles
            normalized_feature_importance_values = model.feature_importances_ / model.feature_importances_.sum()
            return pd.Series(normalized_feature_importance_values, index=features), 'feature_importances_'

        if hasattr(model, 'coef_'):  # Linear models
            coef = np.abs(model.coef_.flatten())
            coef = coef / coef.sum()
            return pd.Series(coef, index=features), 'coef_'
    except ValueError:
        # in case pipeline had an encoder
        pass

    return None, None


@lru_cache(maxsize=32)
def _calc_permutation_importance(
    model: t.Any,
    dataset: 'base.Dataset',
    n_repeats: int = 30,
    mask_high_variance_features: bool = False,
    random_state: int = 42,
    n_samples: int = 10_000,
    alternative_scorer: t.Optional[DeepcheckScorer] = None,
    timeout: int = None
) -> pd.Series:
    """Calculate permutation feature importance. Return nonzero value only when std doesn't mask signal.

    Parameters
    ----------
    model : t.Any
        A fitted model
    dataset : base.Dataset
        dataset used to fit the model
    n_repeats : int , default: 30
        Number of times to permute a feature
    mask_high_variance_features : bool , default: False
        If true, features for which calculated permutation importance values
        varied greatly would be returned has having 0 feature importance
    random_state : int , default: 42
        Random seed for permutation importance calculation.
    n_samples : int , default: 10_000
        The number of samples to draw from X to compute feature importance
        in each repeat (without replacement).
    alternative_scorer : t.Optional[DeepcheckScorer] , default: None

    Returns
    -------
    pd.Series
        feature importance normalized to 0-1 indexed by feature names
    """
    if dataset.label_name is None:
        raise errors.DatasetValidationError("Expected dataset with label.")

    dataset_sample = dataset.sample(n_samples, drop_na_label=True, random_state=random_state)

    # Test score time on the dataset sample
    if alternative_scorer:
        scorer = alternative_scorer
    else:
        task_type = task_type_check(model, dataset)
        default_scorers = get_default_scorers(task_type)
        scorer_name = next(iter(default_scorers))
        single_scorer_dict = {scorer_name: default_scorers[scorer_name]}
        scorer = init_validate_scorers(single_scorer_dict, model, dataset, model_type=task_type)[0]

    if timeout is not None:
        start_time = time.time()
        scorer(model, dataset_sample)
        calc_time = time.time() - start_time

        if calc_time * n_repeats * len(dataset.features) > timeout:
            raise errors.DeepchecksTimeoutError('Permutation importance calculation was not projected to finish in'
                                                f' {timeout} seconds.')
    else:
        warnings.warn('Calculating permutation feature importance without time limit')

    r = permutation_importance(
        model,
        dataset_sample.data[dataset.features],
        dataset_sample.data[dataset.label_name],
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring=scorer.scorer
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

    Parameters
    ----------
    cols_dict : t.Dict[Hashable, t.Any]
        dict where columns are the keys
    dataset : base.Dataset
        dataset used to fit the model
    feature_importances : t.Optional[pd.Series] , default: None
        feature importance normalized to 0-1 indexed by feature names
    n_top : int , default: 10
        amount of columns to show ordered by feature importance (date, index, label are first);
        is used only if model was specified

    Returns
    -------
    Dict
        the dict of columns sorted and limited by feature importance.
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
    """Return the dataframe of columns sorted and limited by feature importance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sort
    ds : base.Dataset
        dataset used to fit the model
    feature_importances : pd.Series
        feature importance normalized to 0-1 indexed by feature names
    n_top : int , default: 10
        amount of columns to show ordered by feature importance (date, index, label are first)
    col : t.Optional[Hashable] , default: None
        name of column to sort the dataframe

    Returns
    -------
    pd.DataFrame
        the dataframe sorted and limited by feature importance.

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

    Parameters
    ----------
    df : pd.DataFrame
        dataframe for which to infer categorical features
    max_categorical_ratio : float , default: 0.01
    max_categories : int , default: 30
    max_float_categories : int , default: 5
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

    Parameters
    ----------
    column : pd.Series
        The name of the column in the dataframe
    max_categorical_ratio : float , default: 0.01
    max_categories : int , default: 30
    max_float_categories : int , default: 5

    Returns
    -------
    bool
        True if is categorical according to input numbers
    """
    n_unique = column.nunique(dropna=True)
    n_samples = len(column.dropna())

    if is_float_dtype(column):
        return n_unique <= max_float_categories

    return n_unique / n_samples < max_categorical_ratio and n_unique <= max_categories
