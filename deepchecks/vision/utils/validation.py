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
