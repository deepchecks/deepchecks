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
"""objects validation utilities."""
import typing as t
import numpy as np
import pandas as pd
import sklearn
from deepchecks import base # pylint: disable=unused-import, is used in type annotations
from deepchecks import errors
from deepchecks.utils.typing import Hashable


__all__ = [
    'model_type_validation', 
    'ensure_hashable_or_mutable_sequence',
    'validate_model',
    'ensure_dataframe',
    'ensure_dataset',
    'ensure_not_empty_dataframe',
    'ensure_not_empty_dataset',
]


def model_type_validation(model: t.Any):
    """Receive any object and check if it's an instance of a model we support.

    Raises:
        DeepchecksValueError: If the object is not of a supported type
    """
    supported_by_class_name = ('CatBoostClassifier', 'CatBoostRegressor')
    supported_by_class_instance = (sklearn.base.BaseEstimator,)
    if (
        not isinstance(model, supported_by_class_instance)
        and model.__class__.__name__ not in supported_by_class_name
    ):
        raise errors.DeepchecksValueError(
            'Model must inherit from one of supported '
            'models: sklearn.base.BaseEstimator or CatBoost'
        )


def validate_model(dataset: 'base.Dataset', model: t.Any):
    """Check model is able to predict on the dataset.

    Raise:
        DeepchecksValueError: if dataset does not match model
    """
    model_type_validation(model)

    error_message = (
        'In order to evaluate model correctness we need not empty dataset '
        'with the same set of features that was used to fit the model. {0}'
    )

    features = dataset.features_columns
    features_names = set(dataset.features)
    model_features = getattr(model, 'feature_names_in_', None)

    if features is None:
        raise errors.DeepchecksValueError(error_message.format(
            'But function received dataset without feature columns.'
        ))

    if len(features) == 0:
        raise errors.DeepchecksValueError(error_message.format(
            'But function received empty dataset.'
        ))

    try:
        model_features = set(model_features) # type: ignore
        if model_features != features_names:
            raise errors.DeepchecksValueError(error_message.format(
                'But function received dataset with a different set of features.'
            ))
    except (TypeError, ValueError):
        # in case if 'model.feature_names_in_' was None or not iterable
        pass

    try:
        model.predict(features.head(1))
    except Exception as exc:
        raise errors.DeepchecksValueError(
            f'Got error when trying to predict with model on dataset: {str(exc)}'
        )


T = t.TypeVar('T', bound=Hashable)


def ensure_hashable_or_mutable_sequence(
    value: t.Union[T, t.MutableSequence[T]],
    message: str = (
        'Provided value is neither hashable nor mutable '
        'sequence of hashable items. Got {type}')
) -> t.List[T]:
    """Validate that provided value is either hashable or mutable sequence of hashable values."""
    if isinstance(value, Hashable):
        return [value]

    if isinstance(value, t.MutableSequence):
        if len(value) > 0 and not isinstance(value[0], Hashable):
            raise errors.DeepchecksValueError(message.format(
                type=f'MutableSequence[{type(value).__name__}]'
            ))
        return list(value)

    raise errors.DeepchecksValueError(message.format(
        type=type(value).__name__
    ))


def ensure_dataframe(
    obj: object,
    error_message: t.Optional[str] = None,
) -> pd.DataFrame:
    """TODO: add comment"""
    error_message = error_message or 'Provided value of type {value_type} cannot be transformed into dataframe.'
    if isinstance(obj, pd.DataFrame):
        return obj
    elif isinstance(obj, base.Dataset):
        return obj.data
    else:
        raise errors.DeepchecksValueError(
            error_message.format(value_type=type(obj).__name__)
        )


def ensure_not_empty_dataframe(
    obj: object,
    error_messages: t.Optional[t.Dict[str, str]] = None
) -> pd.DataFrame:
    """TODO: add comment"""
    error_messages = error_messages or {}
    error_message = error_messages.get('empty') or 'Provided dataframe is empty!'
    obj = ensure_dataframe(obj, error_messages.get('incorrect_value'))
    if len(obj):
        raise errors.DeepchecksValueError(error_message)
    return obj


def ensure_dataset(
    obj: object,
    error_message: t.Optional[str] = None,
) -> 'base.Dataset':
    """TODO: add comment"""
    error_message = error_message or 'Provided value of type {value_type} cannot be transformed into dataset.'
    
    if isinstance(obj, pd.DataFrame):
        return base.Dataset(obj)
    elif isinstance(obj, np.ndarray) and obj.ndim == 2:
        return base.Dataset(pd.DataFrame(obj))
    elif isinstance(obj, base.Dataset):
        return obj
    
    raise errors.DeepchecksValueError(
        error_message.format(value_type=type(obj).__name__)
    )


def ensure_not_empty_dataset(
    obj: object,
    error_messages: t.Optional[t.Dict[str, str]] = None
) -> 'base.Dataset':
    """TODO: add comment"""
    error_messages = error_messages or {}
    error_message = error_messages.get('empty') or 'Provided dataframe is empty!'
    obj = ensure_dataset(obj, error_messages.get('incorrect_value'))
    if len(obj.data) == 0:
        raise errors.DeepchecksValueError(error_message)
    return obj
