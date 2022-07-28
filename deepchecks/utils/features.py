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
# pylint: disable=inconsistent-quotes
"""Utils module containing feature importance calculations."""

# TODO: move tabular functionality to the tabular sub-package

import time
import typing as t

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_datetime_or_timedelta_dtype, is_float_dtype, is_numeric_dtype
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from deepchecks import tabular
from deepchecks.core import errors
from deepchecks.tabular.metric_utils.scorers import (DeepcheckScorer, get_default_scorers, init_validate_scorers,
                                                     task_type_check)
from deepchecks.tabular.utils.validation import validate_model
from deepchecks.utils.logger import get_logger
from deepchecks.utils.strings import is_string_column
from deepchecks.utils.typing import Hashable
from deepchecks.utils.validation import ensure_hashable_or_mutable_sequence

__all__ = [
    '_calculate_feature_importance',
    'calculate_feature_importance_or_none',
    'column_importance_sorter_dict',
    'column_importance_sorter_df',
    'infer_categorical_features',
    'infer_numerical_features',
    'is_categorical',
    'N_TOP_MESSAGE'
]

N_TOP_MESSAGE = '* showing only the top %s columns, you can change it using n_top_columns param'


def calculate_feature_importance_or_none(
        model: t.Any,
        dataset: t.Union['tabular.Dataset', pd.DataFrame],
        force_permutation: bool = False,
        permutation_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Tuple[t.Optional[pd.Series], t.Optional[str]]:
    """Calculate features effect on the label or None if the input is incorrect.

    Parameters
    ----------
    model : t.Any
        a fitted model
    dataset : t.Union['tabular.Dataset', pd.DataFrame]
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
        fi, calculation_type = _calculate_feature_importance(
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
        get_logger().warning('Features importance was not calculated:\n%s', error)
        return None, None


def _calculate_feature_importance(
        model: t.Any,
        dataset: t.Union['tabular.Dataset', pd.DataFrame],
        force_permutation: bool = False,
        permutation_kwargs: t.Dict[str, t.Any] = None,
) -> t.Tuple[pd.Series, str]:
    """Calculate features effect on the label.

    Parameters
    ----------
    model : t.Any
        a fitted model
    dataset : t.Union['tabular.Dataset', pd.DataFrame]
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
    permutation_kwargs = permutation_kwargs or {}
    permutation_kwargs['random_state'] = permutation_kwargs.get('random_state', 42)
    validate_model(dataset, model)
    permutation_failure = None
    calc_type = None
    importance = None

    if force_permutation:
        if isinstance(dataset, pd.DataFrame):
            raise errors.DeepchecksValueError('Cannot calculate permutation feature importance on a pandas Dataframe. '
                                              'In order to force permutation feature importance, please use the Dataset'
                                              ' object.')
        else:
            importance = _calc_permutation_importance(model, dataset, **permutation_kwargs)
            calc_type = 'permutation_importance'

    # If there was no force permutation, or if it failed while trying to calculate importance,
    # we don't take built-in importance in pipelines because the pipeline is changing the features
    # (for example one-hot encoding) which leads to the inner model features
    # being different than the original dataset features
    if importance is None and not isinstance(model, Pipeline):
        # Get the actual model in case of pipeline
        importance, calc_type = _built_in_importance(model, dataset)
        # If found importance and was force permutation failure before, show warning
        if importance is not None and permutation_failure:
            get_logger().warning(permutation_failure)

    # If there was no permutation failure and no importance on the model, using permutation anyway
    if importance is None and permutation_failure is None and isinstance(dataset, tabular.Dataset):
        if not permutation_kwargs.get('skip_messages', False):
            if isinstance(model, Pipeline):
                pre_text = 'Cannot use model\'s built-in feature importance on a Scikit-learn Pipeline,'
            else:
                pre_text = 'Could not find built-in feature importance on the model,'
            get_logger().warning('%s using permutation feature importance calculation instead', pre_text)

        importance = _calc_permutation_importance(model, dataset, **permutation_kwargs)
        calc_type = 'permutation_importance'

    # If after all importance is still none raise error
    if importance is None:
        # FIXME: better message
        raise errors.DeepchecksValueError("Was not able to calculate features importance")
    return importance.fillna(0), calc_type


def _built_in_importance(
        model: t.Any,
        dataset: t.Union['tabular.Dataset', pd.DataFrame],
) -> t.Tuple[t.Optional[pd.Series], t.Optional[str]]:
    """Get feature importance member if present in model."""
    features = dataset.features if isinstance(dataset, tabular.Dataset) else dataset.columns

    if hasattr(model, 'feature_importances_'):  # Ensembles
        if model.feature_importances_ is None:
            return None, None
        normalized_feature_importance_values = model.feature_importances_ / model.feature_importances_.sum()
        return pd.Series(normalized_feature_importance_values, index=features), 'feature_importances_'

    if hasattr(model, 'coef_'):  # Linear models
        if model.coef_ is None:
            return None, None
        coef = np.abs(model.coef_.flatten())
        coef = coef / coef.sum()
        return pd.Series(coef, index=features), 'coef_'

    return None, None


def _calc_permutation_importance(
        model: t.Any,
        dataset: 'tabular.Dataset',
        n_repeats: int = 30,
        mask_high_variance_features: bool = False,
        random_state: int = 42,
        n_samples: int = 10_000,
        alternative_scorer: t.Optional[DeepcheckScorer] = None,
        skip_messages: bool = False,
        timeout: int = None
) -> pd.Series:
    """Calculate permutation feature importance. Return nonzero value only when std doesn't mask signal.

    Parameters
    ----------
    model: t.Any
        A fitted model
    dataset: tabular.Dataset
        dataset used to fit the model
    n_repeats: int, default: 30
        Number of times to permute a feature
    mask_high_variance_features : bool , default: False
        If true, features for which calculated permutation importance values
        varied greatly would be returned has having 0 feature importance
    random_state: int, default: 42
        Random seed for permutation importance calculation.
    n_samples: int, default: 10_000
        The number of samples to draw from X to compute feature importance
        in each repeat (without replacement).
    alternative_scorer: t.Optional[DeepcheckScorer], default: None
        Scorer to use for evaluation of the model performance in the permutation_importance function. If not defined,
        the default deepchecks scorers are used.
    skip_messages: bool, default: False
        If True will not print any message related to timeout or calculation.
    timeout: int, default: None
        Allowed runtime of permutation_importance, in seconds. As we can't limit the actual runtime of the function,
        the timeout parameter is used for estimation of the runtime, done be measuring the inference time of the model
        and multiplying it by number of repeats and features. If the expected runtime is bigger than timeout, the
        calculation is skipped.

    Returns
    -------
    pd.Series
        feature importance normalized to 0-1 indexed by feature names
    """
    if not dataset.has_label():
        raise errors.DatasetValidationError("Expected dataset with label.")

    if len(dataset.features) == 1:
        return pd.Series([1], index=dataset.features)

    dataset_sample = dataset.sample(n_samples, drop_na_label=True, random_state=random_state)

    # Test score time on the dataset sample
    if alternative_scorer:
        scorer = alternative_scorer
    else:
        task_type = task_type_check(model, dataset)
        default_scorers = get_default_scorers(task_type)
        scorer_name = next(iter(default_scorers))
        single_scorer_dict = {scorer_name: default_scorers[scorer_name]}
        scorer = init_validate_scorers(single_scorer_dict, model, dataset)[0]

    start_time = time.time()
    scorer(model, dataset_sample)
    calc_time = time.time() - start_time

    predicted_time_to_run = int(np.ceil(calc_time * n_repeats * len(dataset.features)))

    if timeout is not None:
        if predicted_time_to_run > timeout:
            raise errors.DeepchecksTimeoutError(
                f'Skipping permutation importance calculation: calculation was projected to finish in '
                f'{predicted_time_to_run} seconds, but timeout was configured to {timeout} seconds')
        elif not skip_messages:
            get_logger().info('Calculating permutation feature importance. Expected to finish in %s seconds',
                              predicted_time_to_run)
    elif not skip_messages:
        get_logger().warning('Calculating permutation feature importance without time limit. Expected to finish in '
                             '%s seconds', predicted_time_to_run)

    r = permutation_importance(
        model,
        dataset_sample.features_columns,
        dataset_sample.label_col,
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

    feature_importance = r.importances_mean * significance_mask
    total = feature_importance.sum()

    if total != 0:
        feature_importance = feature_importance / total

    return pd.Series(feature_importance, index=dataset.features)


def get_importance(name: str, feature_importances: pd.Series, ds: 'tabular.Dataset') -> int:
    """Return importance based on feature importance or label/date/index first."""
    if name in feature_importances.keys():
        return feature_importances[name]
    elif ds.has_label() and name == ds.label_name:
        return 1
    elif name in [ds.datetime_name, ds.index_name]:
        return 1
    return 0


def column_importance_sorter_dict(
        cols_dict: t.Dict[Hashable, t.Any],
        dataset: 'tabular.Dataset',
        feature_importances: t.Optional[pd.Series] = None,
        n_top: int = 10
) -> t.Dict:
    """Return the dict of columns sorted and limited by feature importance.

    Parameters
    ----------
    cols_dict : t.Dict[Hashable, t.Any]
        dict where columns are the keys
    dataset : tabular.Dataset
        dataset used to fit the model
    feature_importances : t.Optional[pd.Series] , default: None
        feature importance normalized to 0-1 indexed by feature names
    n_top : int , default: 10
        amount of columns to show ordered by feature importance (date, index, label are first)

    Returns
    -------
    Dict
        the dict of columns sorted and limited by feature importance.
    """
    feature_importances = {} if feature_importances is None else feature_importances

    def key(name):
        return get_importance(name[0], feature_importances, dataset)
    cols_dict = dict(sorted(cols_dict.items(), key=key, reverse=True))
    if n_top:
        return dict(list(cols_dict.items())[:n_top])
    return cols_dict


def column_importance_sorter_df(
        df: pd.DataFrame,
        ds: 'tabular.Dataset',
        feature_importances: pd.Series,
        n_top: int = 10,
        col: t.Optional[Hashable] = None
) -> pd.DataFrame:
    """Return the dataframe of columns sorted and limited by feature importance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sort
    ds : tabular.Dataset
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
    if len(df) == 0:
        return df

    feature_importances = {} if feature_importances is None else feature_importances

    def key(column):
        return [get_importance(name, feature_importances, ds) for name in column]
    if col:
        df = df.sort_values(by=[col], key=key, ascending=False)
    df = df.sort_index(key=key, ascending=False)
    if n_top:
        return df.head(n_top)
    return df


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
