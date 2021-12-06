# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""objects validation utilities."""
import typing as t
import sklearn
from deepchecks import errors
from deepchecks.utils.typing import Hashable


__all__ = ['model_type_validation', 'ensure_hashable_or_mutable_sequence']


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


T = t.TypeVar('T', bound=Hashable)


def ensure_hashable_or_mutable_sequence(
    value: t.Union[T, t.MutableSequence[T]],
    message: str = (
        'Provided value is neither hashable nor mutable '
        'sequence of hashable items!. Got {type}')
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
