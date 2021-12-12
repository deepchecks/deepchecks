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
from numbers import Number

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer, make_scorer, accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.base import ClassifierMixin, RegressorMixin

from deepchecks import base #pylint: disable=unused-import; it is used for type annotations
from deepchecks import errors
from deepchecks.utils import validation


__all__ = [
    'ModelType',
    'task_type_check',
    'get_scorers_list',
    'validate_scorer',
    'DEFAULT_SCORERS_DICT',
    'DEFAULT_SINGLE_SCORER',
    'get_scores_ratio'
]


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


DEFAULT_REGRESSION_SCORERS = {
    'RMSE': make_scorer(mean_squared_error, squared=False, greater_is_better=False),
    'MSE': make_scorer(mean_squared_error, greater_is_better=False),
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
        return ModelType.REGRESSION
    else:
        labels = t.cast(pd.Series, dataset.label_col)
        unique_labels = labels.unique()

        if sorted(unique_labels) != list(range(len(unique_labels))):
            raise errors.DeepchecksValueError(
                'Classification labels must be a consecutive set from 0 to MAX_LABEL,'
                f' found {sorted(unique_labels)}.'
            )

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
    if not task_type in expected_types:
        raise errors.DeepchecksValueError(
            f'Expected model to be a type from {[e.value for e in expected_types]}, '
            f'but received model of type: {task_type.value}'
        )


def get_scorers_list(
    model,
    dataset: 'base.Dataset',
    alternative_scorers: t.Dict[str, t.Callable] = None
) -> t.Dict[str, t.Callable]:
    """Return list of scorer objects to use in a score-dependant check.

    If no alternative_scorers is supplied, then a default list of scorers is used per task type, as it is inferred
    from the dataset and model. If a list is supplied, then the scorer functions are checked and used instead.

    Args:
        model (BaseEstimator): Model object for which the scores would be calculated
        dataset (Dataset): Dataset object on which the scores would be calculated
        alternative_scorers (Dict[str, Callable]): Optional dictionary of sklearn scorers to use instead of default list

    Returns:
        Dictionary containing names of scorer functions.
    """
    if alternative_scorers:
        scorers = {}
        for name, scorer in alternative_scorers.items():
            scorers[name] = validate_scorer(scorer, model, dataset)
    else:
        # Check for model type
        model_type = task_type_check(model, dataset)
        scorers = DEFAULT_SCORERS_DICT[model_type]

    return scorers


def validate_scorer(scorer, model, dataset):
    """If string, get scorer from sklearn. If callable, do heuristic to see if valid."""
    # Borrowed code from:
    # https://github.com/scikit-learn/scikit-learn/blob/844b4be24d20fc42cc13b957374c718956a0db39/sklearn/metrics/_scorer.py#L421
    if isinstance(scorer, str):
        return get_scorer(scorer)
    elif callable(scorer):
        # Check that scorer runs for given model and data
        assert isinstance(scorer(model, dataset.data[dataset.features].head(2), dataset.label_col.head(2)),
                          Number)
        return scorer
    else:
        raise errors.DeepchecksValueError(
            f'scorer value should either be a callable or string. '
            f'But {type(scorer).__name__} was passed'
        )


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
