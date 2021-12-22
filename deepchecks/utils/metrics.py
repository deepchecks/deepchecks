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
"""Utils module containing utilities for checks working with metrics."""
import typing as t
import enum
from functools import partial
from numbers import Number

import numpy as np
import pandas as pd

from sklearn.metrics import get_scorer, make_scorer, accuracy_score, precision_score, recall_score, \
    mean_squared_error, f1_score, mean_absolute_error
from sklearn.base import ClassifierMixin, RegressorMixin

from deepchecks import base  # pylint: disable=unused-import; it is used for type annotations
from deepchecks import errors
from deepchecks.utils import validation


__all__ = [
    'ModelType',
    'task_type_check',
    'get_scorers_dict',
    'initialize_single_scorer',
    'DEFAULT_SCORERS_DICT',
    'DEFAULT_SINGLE_SCORER',
    'MULTICLASS_SCORERS_NON_AVERAGE',
    'get_scores_ratio',
    'initialize_user_scorers',
    'get_scorer_single'
]

from deepchecks.utils.strings import is_string_column


class ModelType(enum.Enum):
    """Enum containing suppoerted task types."""

    REGRESSION = 'regression'
    BINARY = 'binary'
    MULTICLASS = 'multiclass'


DEFAULT_BINARY_SCORERS = {
    'Accuracy': make_scorer(accuracy_score),
    'Precision': make_scorer(precision_score, zero_division=0),
    'Recall': make_scorer(recall_score, zero_division=0)
}


DEFAULT_MULTICLASS_SCORERS = {
    'Accuracy': make_scorer(accuracy_score),
    'Precision - Macro Average': make_scorer(precision_score, average='macro', zero_division=0),
    'Recall - Macro Average': make_scorer(recall_score, average='macro', zero_division=0)
}

MULTICLASS_SCORERS_NON_AVERAGE = {
    'F1': make_scorer(f1_score, average=None),
    'Precision': make_scorer(precision_score, average=None),
    'Recall': make_scorer(recall_score, average=None)
}


DEFAULT_REGRESSION_SCORERS = {
    'RMSE': make_scorer(mean_squared_error, squared=False, greater_is_better=False),
    'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
}


DEFAULT_SINGLE_SCORER = {
    ModelType.BINARY: 'Accuracy',
    ModelType.MULTICLASS: 'Accuracy',
    ModelType.REGRESSION: 'RMSE'
}


DEFAULT_SCORERS_DICT = {
    ModelType.BINARY: DEFAULT_BINARY_SCORERS,
    ModelType.MULTICLASS: DEFAULT_MULTICLASS_SCORERS,
    ModelType.REGRESSION: DEFAULT_REGRESSION_SCORERS
}


def task_type_check(
    model: t.Union[ClassifierMixin, RegressorMixin],
    dataset: 'base.Dataset'
) -> ModelType:
    """Check task type (regression, binary, multiclass) according to model object and label column.

    Args:
        model (Union[ClassifierMixin, RegressorMixin]): Model object - used to check if has predict_proba()
        dataset (Dataset): dataset - used to count the number of unique labels

    Returns:
        TaskType enum corresponding to the model and dataset
    """
    validation.model_type_validation(model)
    dataset.validate_label()

    if not hasattr(model, 'predict_proba'):
        if is_string_column(dataset.label_col):
            raise errors.DeepchecksValueError(
                'Model was identified as a regression model, but label column was found to contain strings.'
            )
        else:
            return ModelType.REGRESSION
    else:
        labels = t.cast(pd.Series, dataset.label_col)

        return (
            ModelType.MULTICLASS
            if labels.nunique() > 2
            else ModelType.BINARY
        )


def task_type_validation(
    model: t.Union[ClassifierMixin, RegressorMixin],
    dataset: 'base.Dataset',
    expected_types: t.Sequence[ModelType]
):
    """Validate task type (regression, binary, multiclass) according to model object and label column.

    Args:
        model (Union[ClassifierMixin, RegressorMixin]): Model object - used to check if has predict_proba()
        dataset (Dataset): dataset - used to count the number of unique labels
        expected_types (List[ModelType]): allowed types of model
        check_name (str): check name to print in error

    Raises:
            DeepchecksValueError if model type doesn't match one of the expected_types
    """
    task_type = task_type_check(model, dataset)
    if task_type not in expected_types:
        raise errors.DeepchecksValueError(
            f'Expected model to be a type from {[e.value for e in expected_types]}, '
            f'but received model of type: {task_type.value}'
        )


def calculate_scorer_with_nulls(model, dataset: 'base.Dataset', scorer: t.Callable) -> float:
    label = dataset.label_col
    valid_idx = label.notna()
    return scorer(model, dataset.features_columns[valid_idx], label[valid_idx])


def get_scorers_dict(
    model,
    dataset: 'base.Dataset',
    alternative_scorers: t.Dict[str, t.Callable] = None,
    multiclass_avg: bool = True
) -> t.Dict[str, t.Callable]:
    """Return list of scorer objects to use in a score-dependant check.

    If no alternative_scorers is supplied, then a default list of scorers is used per task type, as it is inferred
    from the dataset and model. If a list is supplied, then the scorer functions are checked and used instead.

    Args:
        model (BaseEstimator): Model object for which the scores would be calculated
        dataset (Dataset): Dataset object on which the scores would be calculated
        alternative_scorers (Dict[str, Callable]): Optional dictionary of sklearn scorers to use instead of default list
        multiclass_avg

    Returns:
        Dictionary containing names of scorer functions.
    """
    # Check for model type
    model_type = task_type_check(model, dataset)

    if alternative_scorers:
        scorers = alternative_scorers
    else:
        if model_type == ModelType.MULTICLASS and multiclass_avg is False:
            scorers = MULTICLASS_SCORERS_NON_AVERAGE
        else:
            scorers = DEFAULT_SCORERS_DICT[model_type]

    for name, scorer in scorers.items():
        should_return_array = model_type == ModelType.MULTICLASS and multiclass_avg is False
        validate_scorer(scorer, model, dataset, should_return_array, name)

    # Transform scorers into calculate_without_nulls
    return {k: partial(calculate_scorer_with_nulls, scorer=v) for k, v in scorers.items()}


def get_scorer_single(model, dataset: 'base.Dataset', alternative_scorer: t.Tuple[str, t.Callable] = None,
                      multiclass_avg: bool = True):
    """Return single score to use in check, and validate scorer fit the model and dataset."""
    model_type = task_type_check(model, dataset)
    multiclass_array = model_type == ModelType.MULTICLASS and multiclass_avg is False

    if alternative_scorer is not None:
        scorer_name = alternative_scorer[0]
        scorer_func = alternative_scorer[1]
    else:
        if multiclass_array:
            scorer_name = 'F1'
            scorer_func = MULTICLASS_SCORERS_NON_AVERAGE[scorer_name]
        else:
            scorer_name = DEFAULT_SINGLE_SCORER[model_type]
            scorer_func = DEFAULT_SCORERS_DICT[model_type][scorer_name]

    validate_scorer(scorer_func, model, dataset, multiclass_array, scorer_name)
    # Transform scorer into calculate without nulls
    return scorer_name, partial(calculate_scorer_with_nulls, scorer=scorer_func)


def initialize_single_scorer(scorer: t.Optional[t.Union[str, t.Callable]], scorer_name=None) \
        -> t.Optional[t.Tuple[str, t.Callable]]:
    """If string, get scorer from sklearn. If none, return none."""
    if scorer is None:
        return None

    scorer_name = scorer_name or (scorer if isinstance(scorer, str) else 'User Scorer')
    if isinstance(scorer, str):
        return scorer_name, get_scorer(scorer)
    elif callable(scorer):
        return scorer_name, scorer
    else:
        scorer_type = type(scorer).__name__
        if scorer_name:
            message = f'Scorer {scorer_name} value should be either a callable or string but got: {scorer_type}'
        else:
            message = f'Scorer should be should be either a callable or string but got: {scorer_type}'
        raise errors.DeepchecksValueError(message)


def validate_scorer(scorer: t.Callable, model, dataset, should_return_array: bool, scorer_name: str):
    """Validate given scorer for the model and dataset."""
    label = dataset.label_col
    valid_idx = label.notna()
    result = scorer(model, dataset.features_columns[valid_idx].head(2), label[valid_idx].head(2))
    if should_return_array:
        if not isinstance(result, np.ndarray):
            raise errors.DeepchecksValueError(f'Expected scorer {scorer_name} to return np.ndarray '
                                              f'but got: {type(result).__name__}')
        expected_types = t.cast(
            str,
            np.typecodes['AllInteger'] + np.typecodes['AllFloat']  # type: ignore
        )
        kind = result.dtype.kind
        if kind not in expected_types:
            raise errors.DeepchecksValueError(f'Expected scorer {scorer_name} to return np.ndarray of number kind '
                                              f'but got: {kind}')
    else:
        if not isinstance(result, Number):
            raise errors.DeepchecksValueError(f'Expected scorer {scorer_name} to return number '
                                              f'but got: {type(result).__name__}')


def initialize_user_scorers(alternative_scorers: t.Optional[t.Mapping[str, t.Callable]]) -> \
        t.Optional[t.Dict[str, t.Callable]]:
    """Initialize user scorers and return all of them as callable."""
    if alternative_scorers is None:
        return None
    elif len(alternative_scorers) == 0:
        raise errors.DeepchecksValueError('Scorers dictionary can\'t be empty')
    else:
        return dict([initialize_single_scorer(v, scorer_name=k) for k, v in alternative_scorers.items()])


def get_scores_ratio(train_score: float, test_score: float, max_ratio=np.Inf) -> float:
    """Return the ratio of test metric compared to train metric."""
    if train_score == 0:
        return max_ratio
    else:
        ratio = test_score / train_score
        if train_score < 0 and test_score < 0:
            ratio = 1 / ratio
        ratio = min(max_ratio, ratio)
        return ratio
