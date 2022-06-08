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
"""Tabular objects validation utilities."""
import typing as t

import numpy as np
import pandas as pd

from deepchecks import tabular
from deepchecks.core import errors
from deepchecks.utils.typing import BasicModel

__all__ = [
    'model_type_validation',
    'validate_model',
    'ensure_dataframe_type',
    'ensure_predictions_shape',
    'ensure_predictions_proba',
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
    data: 'tabular.Dataset',
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

    if isinstance(data, tabular.Dataset):
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
    elif isinstance(obj, tabular.Dataset):
        return obj.data
    else:
        raise errors.DeepchecksValueError(
            f'dataset must be of type DataFrame or Dataset, but got: {type(obj).__name__}'
        )


def ensure_predictions_shape(pred: np.ndarray, data: pd.DataFrame) -> np.ndarray:
    """Ensure the predictions are in the right shape and if so return them. else raise error."""
    if pred.shape != (len(data), ):
        raise errors.ValidationError(f'Prediction array excpected to be of shape {(len(data), )} '
                                     f'but was: {pred.shape}')
    return pred


def ensure_predictions_proba(pred_proba: np.ndarray, data: pd.DataFrame) -> np.ndarray:
    """Ensure the predictions are in the right shape and if so return them. else raise error."""
    if len(pred_proba) != len(data):
        raise errors.ValidationError(f'Prediction propabilities excpected to be of length {len(data)} '
                                     f'but was: {len(pred_proba)}')
    return pred_proba
