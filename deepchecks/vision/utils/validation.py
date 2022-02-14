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
"""Module for validation of the vision module."""
import typing as t

import torch
from deepchecks.core import errors
from deepchecks import vision  # pylint: disable=unused-import, is used in type annotations


__all__ = ['validate_model', 'apply_to_tensor']


def validate_model(dataset: 'vision.VisionData', model: t.Any):
    """Receive a dataset and a model and check if they are compatible.

    Parameters
    ----------
    dataset : VisionData
        Built on a dataloader on which the model can infer.
    model : Any
        Model to be validated

    Raises
    ------
    DeepchecksValueError
        If the dataset and the model are not compatible
    """
    try:
        model(next(iter(dataset.get_data_loader()))[0])
    except Exception as exc:
        raise errors.ModelValidationError(
            f'Got error when trying to predict with model on dataset: {str(exc)}'
        )


T = t.TypeVar('T')


def apply_to_tensor(
    x: T,
    fn: t.Callable[[torch.Tensor], torch.Tensor]
) -> T:
    """Apply provided function to tensor instances recursivly."""
    if isinstance(x, torch.Tensor):
        return t.cast(T, fn(x))
    elif isinstance(x, (str, bytes, bytearray)):
        return x
    elif isinstance(x, (list, tuple, set)):
        return type(x)(apply_to_tensor(it, fn) for it in x)
    elif isinstance(x, dict):
        return type(x)((k, apply_to_tensor(v, fn)) for k, v in x.items())
    return x
