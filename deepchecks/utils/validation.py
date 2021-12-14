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

import sklearn

from deepchecks import base # pylint: disable=unused-import, is used in type annotations
from deepchecks import errors
from deepchecks.utils.typing import Hashable


__all__ = ['model_type_validation', 'ensure_hashable_or_mutable_sequence', 'validate_model']


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
            'models: sklearn.base.BaseEstimator or CatBoost, '
            f'Recived: {model.__class__.__name__}'
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
