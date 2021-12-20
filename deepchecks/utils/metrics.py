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
    'get_scorers',
    'get_validate_scorer',
    'get_metrics_ratio',
    'initialize_user_scorers'
]

from deepchecks.errors import DeepchecksNotSupportedError, DeepchecksValueError


class ModelType(enum.Enum):
    """Enum containing suppoerted task types."""

    REGRESSION = 'regression'
    BINARY = 'binary'
    MULTICLASS = 'multiclass'


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


def initialize_user_scorers(alternative_scorers: t.Mapping[str, t.Union[str, t.Callable]]):
    """Transform and validate user given scorers dict."""
    if len(alternative_scorers) == 0:
        raise DeepchecksValueError('can\'t have empty scorers dictionary')

    scorers = {}
    for name, scorer in alternative_scorers.items():
        scorers[name] = get_single_scorer(scorer)

    return scorers


def get_scorers(
    model,
    dataset: 'base.Dataset',
    user_scorers: t.Dict[str, t.Callable] = None,
    single: bool = False,
    average: bool = True
) -> t.Dict[str, t.Callable]:
    """Return list of scorer objects to use in a metrics-dependant check.

    If no alternative_metrics is supplied, then a default list of metrics is used per task type, as it is inferred
    from the dataset and model. If a list is supplied, then the scorer functions are checked and used instead.

    Args:
        model (BaseEstimator): Model object for which the metrics would be calculated
        dataset (Dataset): Dataset object on which the metrics would be calculated
        user_scorers (Dict[str, Callable]): Optional dictionary of sklearn scorers to use instead of default list
        single (bool): Whether to return a single default scorer
        average (bool): For multiclass, whether to do average to do over the classes. False will return each class
            in a numpy array.
    Returns:
        Dictionary containing names of metrics and scorer functions for the metrics.
    """
    # Check for model type
    model_type = task_type_check(model, dataset)

    if user_scorers:
        scorers = user_scorers
    else:
        if model_type == ModelType.BINARY:
            scorers = {
                'Accuracy': make_scorer(accuracy_score)
            }
            if not single:
                scorers.update({
                    'Precision': make_scorer(precision_score, zero_division=0),
                    'Recall': make_scorer(recall_score, zero_division=0)
                })
        elif model_type == ModelType.MULTICLASS:
            if average:
                scorers = {
                    'Accuracy': make_scorer(accuracy_score)
                }
                if not single:
                    scorers.update({
                        'Precision - Macro Average': make_scorer(precision_score, average='macro', zero_division=0),
                        'Recall - Macro Average': make_scorer(recall_score, average='macro', zero_division=0)
                    })
            else:
                scorers = {
                    'Precision': make_scorer(precision_score, average=None, zero_division=0),
                }
                if not single:
                    scorers.update({
                        'Recall': make_scorer(recall_score, average=None, zero_division=0)
                    })
        elif model_type == ModelType.REGRESSION:
            scorers = {
                'RMSE': make_scorer(mean_squared_error, squared=False, greater_is_better=False)
            }
            if not single:
                scorers.update({
                    'MSE': make_scorer(mean_squared_error, greater_is_better=False)
                })
        else:
            raise DeepchecksNotSupportedError('Default scorers not supported for given model type')

    # Validate
    for scorer in scorers.values():
        result_is_array = model_type == ModelType.MULTICLASS and average is False
        validate_scorer(scorer, model, dataset, result_is_array)

    return scorers


def validate_scorer(scorer, model, dataset, result_is_array=False):
    """Validate scorer works for dataset and model."""
    try:
        result = scorer(model, dataset.data[dataset.features].head(2), dataset.label_col.head(2))
    except Exception as exc:
        raise DeepchecksValueError(f'Error using scorer: {str(exc)}') from exc

    if not result_is_array and not isinstance(result, Number):
        raise DeepchecksValueError(f'Expected scorer to return number but got: {type(result).__name__}')

    if result_is_array:
        expected_types = t.cast(
            str,
            np.typecodes['AllInteger'] + np.typecodes['AllFloat']  # type: ignore
        )
        if not isinstance(result, np.ndarray):
            raise DeepchecksValueError(f'Expected scorer to return numpy array, but got: {type(result).__name__}')
        if result.dtype.kind not in expected_types:
            raise DeepchecksValueError(f'Expected scorer to return ndarray of type int/float, '
                                       f'but got ndarray of type: {result.dtype.kind}')


def get_validate_scorer(scorer, model, dataset):
    """Get scorer if string or callable and validate it works on data and model."""
    scorer = get_single_scorer(scorer)
    validate_scorer(scorer, model, dataset)
    return scorer


def get_single_scorer(scorer: t.Union[str, t.Callable]):
    """Verify scorer is string or callable, and return scorer as callable."""
    if isinstance(scorer, t.Callable):
        return scorer
    elif isinstance(scorer, str):
        return get_scorer(scorer)
    else:
        raise DeepchecksValueError(
            f'Expected scorer to be scorer name or callable but was: {type(scorer).__name__}'
        )


def get_metrics_ratio(train_metric: float, test_metric: float, max_ratio=np.Inf) -> float:
    """Return the ratio of test metric compared to train metric."""
    if train_metric == 0:
        return max_ratio
    else:
        ratio = test_metric / train_metric
        if train_metric < 0 and test_metric < 0:
            ratio = 1 / ratio
        ratio = min(max_ratio, ratio)
        return ratio
