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
import pandas as pd

from deepchecks import base  # pylint: disable=unused-import, is used in type annotations
from deepchecks import errors
from deepchecks.utils.typing import Hashable, BasicModel


__all__ = [
    'model_type_validation',
    'ensure_hashable_or_mutable_sequence',
    'validate_model',
    'ensure_dataframe_type'
]

supported_models_link = ('https://docs.deepchecks.com/en/stable/user-guide/supported_models.html'
                         '?utm_source=display_output&utm_medium=referral&utm_campaign=exception_link')
supported_models_html = f'<a href="{supported_models_link}" target="_blank">supported model types</a>'


def model_type_validation(model: t.Any):
    """Receive any object and check if it's an instance of a model we support.

    Parameters
    ----------
    model: t.Any
    Raises
    ------
    DeepchecksValueError
        If the object is not of a supported type
    """
    if not isinstance(model, BasicModel):
        raise errors.ModelValidationError(
            f'Model supplied does not meets the minimal interface requirements. Read more about {supported_models_html}'
        )


def validate_model(
    data: t.Union['base.Dataset', pd.DataFrame],
    model: t.Any
):
    """Check model is able to predict on the dataset.

    Parameters
    ----------
    data : t.Union['base.Dataset', pd.DataFrame]
    model : t.Any
    Raises
    ------
    DeepchecksValueError
        if dataset does not match model.
    """
    error_message = (
        'In order to evaluate model correctness we need not empty dataset '
        'with the same set of features that was used to fit the model. {0}'
    )

    if isinstance(data, base.Dataset):
        features = data.data[data.features]
    else:
        features = data

    if features is None:
        raise errors.DeepchecksValueError(error_message.format(
            'But function received dataset without feature columns.'
        ))

    if len(features) == 0:
        raise errors.DeepchecksValueError(error_message.format(
            'But function received empty dataset.'
        ))

    try:
        model.predict(features.head(1))
    except Exception as exc:
        raise errors.ModelValidationError(
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


def ensure_dataframe_type(obj: t.Any) -> pd.DataFrame:
    """Ensure that given object is of type DataFrame or Dataset and return it as DataFrame. else raise error.

    Parameters
    ----------
    obj : t.Any
        Object to ensure it is DataFrame or Dataset
    Returns
    -------
    pd.DataFrame
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    elif isinstance(obj, base.Dataset):
        return obj.data
    else:
        raise errors.DeepchecksValueError(
            f'dataset must be of type DataFrame or Dataset, but got: {type(obj).__name__}'
        )
