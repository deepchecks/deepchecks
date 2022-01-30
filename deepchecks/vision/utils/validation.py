"""Module for validation of the vision module."""
import typing as t

from torch import nn

from deepchecks.core import errors


def model_type_validation(model: t.Any):
    """Receive any object and check if it's an instance of a model we support.

    Parameters
    ----------
    model : Any
        Any object to be checked.

    Raises
    ------
    DeepchecksValueError
        If the object is not of a supported type
    """
    if not isinstance(model, nn.Module):
        raise errors.DeepchecksValueError(
            'Model must inherit from torch.nn.Module '
            f'Received: {model.__class__.__name__}'
        )
